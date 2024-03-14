import os
import numpy as np
import torch
import argparse
from modules.model import Network
from modules.dataset import Embedding_dataset
from modules.loss import InstanceLoss, ClusterLoss
from modules.train import train_epoch, valid_epoch
from modules.cluster import inference
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from utils.seed import seed_everything
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.optim import Adam, AdamW
from utils.parser import load_args, load_yaml_param_settings
import wandb
from torch.utils.data import Dataset, DataLoader, random_split
from transformers.optimization import get_linear_schedule_with_warmup
from glob import glob
import multiprocessing
from multiprocessing import set_start_method, Process
import asyncio
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial
import torch.multiprocessing as mp
from datetime import datetime
import yaml
import logging

class Get_Data():
    def __init__(self, directory, save_path=None):
        self.directory = directory
        self.save_path = save_path
        
    def get_file_list(self):
        self.list = glob(f"{self.directory}/*.csv")
        if self.save_path is not None:
            self.list = [file for file in self.list if not os.path.exists(f"{self.save_path}/{file.split('/')[-1]}")]

        self.list.sort()
        # * Filtering files
        # self.list = [csv for csv in self.list if int(csv.split('.')[0].split('-')[0]) >= 1990]
        return self.list
    
class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.epochs = self.config['train']['epochs']
        self.batch_size = self.config['train']['batch_size']
        self.lr = self.config['train']['lr']
        self.saving_path = self.config['train']['saving_path']
        self.model_saving_strategy = self.config['train']['model_saving_strategy']
        self.seed = self.config['train']['seed']
        self.num_workers = self.config['train']['num_workers']
        self.device = self.config['train']['device']
        self.epochs = self.config['train']['epochs']
        self.use_accelerator = self.config['train']['use_accelerator']
        self.pin_memory = self.config['train']['pin_memory']
        # Seed everything
        seed_everything(self.seed)

    def load_data(self, file_path):
        # 데이터 로딩 로직
        train_data_frame = pd.read_csv(file_path)

        # Drop firm name
        train_data = train_data_frame.iloc[:, 1:]
        # Load data
        train_data = Embedding_dataset(self.config, data=train_data)

        train_loader = DataLoader(train_data, 
                                batch_size=self.batch_size, 
                                shuffle=True, 
                                num_workers=self.num_workers, 
                                drop_last=True, 
                                pin_memory= self.pin_memory,
                                persistent_workers=True) #! persistent_workers=True: worker를 메모리에 유지
        

        return train_loader
    
    def train(self, file_path):
        
        # Load data
        train_loader = self.load_data(file_path)

        # Load model
        model = Network(cluster_num = self.config['model']['cluster_num'],
                        dim_in_out = self.config['model']['n_bins'],
                        num_features = self.config['model']['num_features'],
                        hidden_dim = self.config['model']['hidden_dim'],
                        depth = self.config['model']['depth'],
                        heads = self.config['model']['heads'],
                        pre_norm = self.config['model']['pre_norm'],
                        use_simple_rmsnorm = self.config['model']['use_simple_rmsnorm']
        )
        optimizer = AdamW(model.parameters(), lr = self.lr)
        num_training_steps = self.epochs * len(train_loader)  # 전체 학습에 걸쳐서 수행되는 step 수
        
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps= self.config['train']['warmup_steps'], 
                                                    num_training_steps=num_training_steps)
            
        criterion_ins = InstanceLoss()
        criterion_clu = ClusterLoss()

        # Accelerator 🤗설정
        accelerator = Accelerator(log_with='wandb' if self.config['train']['use_wandb'] else None)
        
        # WandB 로깅 설정
        if self.config['train']['use_wandb']:
            wandb_config = {**self.config['train'], **self.config['model']}
            if accelerator.is_local_main_process:
                wandb.init(project="Pairs_trading", config=self.config)
            accelerator.init_trackers("Pairs_trading", config=wandb_config)

        # TODO: accelerator 조절할 필요 있음
        if self.use_accelerator:
            device = accelerator.device
            model, criterion_ins, criterion_clu, train_loader, optimizer, scheduler = accelerator.prepare(
                model, criterion_ins, criterion_clu, train_loader, optimizer, scheduler
            )
        else:
            device = self.device
            model, criterion_ins, criterion_clu, train_loader, optimizer, scheduler = \
                model.to(device), criterion_ins.to(device), criterion_clu.to(device), train_loader, optimizer, scheduler

        # Finally train
        best_loss = 1e10
        best_model_dict = None
        training_step = 0
        
        epochs = self.epochs
        max_train_steps = epochs * len(train_loader)
        progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process, position=1)
        
        for epoch in range(epochs):
            
            model, train_loss, training_step = train_epoch(
                model,
                criterion_ins,
                criterion_clu,
                train_loader,
                optimizer,
                scheduler,
                device,
                epoch,
                training_step,
                progress_bar,
                accelerator,
                self.use_accelerator,
                self.config['train']['use_wandb']
            )
            
            # Log
            if self.config['train']['use_wandb']:
                accelerator.log({"train_loss": train_loss})
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")
        
        self.model = model

        if self.use_accelerator:
            accelerator.wait_for_everyone()
            self.model = accelerator.unwrap_model(model)
        state_dict = model.state_dict()
        
        if self.config['train']['use_wandb']:
            accelerator.end_training()

    def cluster_inference(self, file_path):
         #* 동일한 데이터를 사용한다.
        test_data_frame = pd.read_csv(file_path)
        test_data = test_data_frame.iloc[:, 1:]
        test_data = Embedding_dataset(self.config, data=test_data, is_train=False)
        test_loader = DataLoader(test_data, 
                                 batch_size=self.batch_size, 
                                 shuffle=False, # Inference에서는 shuffle을 False로 설정해야 함
                                 num_workers=self.num_workers, 
                                 drop_last= False, # Inference에서는 drop_last를 False로 설정해야 함
                                 pin_memory=self.pin_memory)
        
        # self.model.load_state_dict(torch.load(f"{self.saving_path}/best_model_{self.config['model']['cluster_num']}_{file_path}.pt"))
        # self.model.load_state_dict(torch.load(f"{self.saving_path}/{self.config['model']['cluster_num']}_{file_path}.pt"))
        features, prob, clusters = inference(test_loader, self.model, self.device)

        # No outlier
        clusters = clusters + 1
        predictions = pd.DataFrame({'firms': test_data_frame['firms'], 'clusters': clusters, 'mom1': test_data_frame['mom1']})

        ##! save
        if not os.path.exists(f"{self.saving_path}/predictions/{self.config['model']['cluster_num']}_{file_path.split('/')[0]}"):
            os.makedirs(f"{self.saving_path}/predictions/{self.config['model']['cluster_num']}_{file_path.split('/')[0]}")
        # print(self.config['model']['cluster_num'], file_path)
        predictions.to_csv(f"{self.saving_path}/predictions/{self.config['model']['cluster_num']}_{file_path}", index=False)

        #* probability saving
        prob_df = pd.DataFrame(prob, columns=[f'prob_{i}' for i in range(self.config['model']['cluster_num'])])
        prob_df['firms'] = test_data_frame['firms']

        if not os.path.exists(f"{self.saving_path}/prob/{self.config['model']['cluster_num']}_{file_path.split('/')[0]}"):
            os.makedirs(f"{self.saving_path}/prob/{self.config['model']['cluster_num']}_{file_path.split('/')[0]}")
        prob_df.to_csv(f"{self.saving_path}/prob/{self.config['model']['cluster_num']}_{file_path}", index=False)

# ! Multi GPU
def setup(rank):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    torch.cuda.set_device(rank)
    # print(f"Process on GPU: {torch.cuda.current_device()}")
    # logging.info(f"Process on GPU: {torch.cuda.current_device()}")

def process_file(rank, file_path, config):
    setup(rank)

    trainer = ModelTrainer(config)
    trainer.train(file_path)
    trainer.cluster_inference(file_path)

def main(config):
    set_start_method('spawn', force=True)

    world_size = 4  # 사용 가능한 GPU 수
    csv_loader = Get_Data(config['data'], config['data_refine'])
    file_list = csv_loader.get_file_list()

    total_batches = len(file_list) // world_size + (len(file_list) % world_size > 0)
    progress_bar = tqdm(total=total_batches, desc="Processing Files", unit="batch", position=0, leave=True)


    for i in range(0, len(file_list), world_size):
        start_time = datetime.now()

        current_batch = file_list[i:i + world_size]
        processes = []
        for i, file_path in enumerate(current_batch):
            rank = i % world_size
            p = Process(target=process_file, args=(rank, file_path, config))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        end_time = datetime.now()
        time_taken = (end_time - start_time).total_seconds()
        progress_bar.set_postfix(batch_time=f"{time_taken:.2f}s")
        progress_bar.update(1)
        
#############################################################################
# ! Single GPU

# def main(config):
#     csv_loader = Get_Data(config['data'], config['data_refine'])
#     file_list = csv_loader.get_file_list()
#     for file in file_list:
#         trainer = ModelTrainer(config)
#         trainer.train(file)
#         trainer.cluster_inference(file)

if __name__ == "__main__":

    args = load_args()
    config = load_yaml_param_settings(args.config)
    
    # Save the configuration
    if not os.path.exists(config['train']['saving_path']):
        os.makedirs(config['train']['saving_path'])
    date = datetime.now().strftime("%Y%m%d-%H%M")
    with open(f"{config['train']['saving_path']}/run_{date}_config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    main(config)