import os
import gc
import numpy as np
import concurrent.futures
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from modules.model import Network
from modules.dataset import Embedding_dataset
from modules.loss import InstanceLoss, ClusterLoss
from modules.train import train_epoch, valid_epoch
from modules.cluster import inference, inference_parallel
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from utils.seed import seed_everything
from torch.optim import Adam, AdamW
from utils.preprocessing import preprocess_data
from utils.parser import load_args, load_yaml_param_settings
from utils.logger import get_logger
import wandb
from torch.utils.data import Dataset, DataLoader, random_split
from transformers.optimization import get_linear_schedule_with_warmup
from glob import glob
import multiprocessing as mp
from multiprocessing import set_start_method, Process, Pool
import asyncio
from functools import partial
from datetime import datetime
import yaml
import sys
import logging
import argparse

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

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
        self.hidden_dim = self.config['model']['hidden_dim']
        self.n_bins = self.config['model']['n_bins']
        self.std = self.config['model']['augment_std']
        self.mask = self.config['model']['masking_ratio']
        self.cluster_tau = self.config['model']['cluster_tau']
        self.lr = self.config['train']['lr']
        self.saving_path = self.config['train']['saving_path']
        self.model_saving_strategy = self.config['train']['model_saving_strategy']
        self.seed = self.config['train']['seed']
        self.num_workers = self.config['train']['num_workers']
        self.device = self.config['train']['device']
        self.use_accelerator = self.config['train']['use_accelerator']
        self.pin_memory = self.config['train']['pin_memory']
        self.persist_workers = self.config['train']['persist_workers']
        # Seed everything
        seed_everything(self.seed)


    def load_data(self, file_path):
        # 데이터 로딩 로직
        train_data_frame = pd.read_csv(file_path)

        processed_data = preprocess_data(train_data_frame, file_path, logger)

        # Load data
        train_data = Embedding_dataset(self.config, data=processed_data)
        train_loader = DataLoader(train_data, 
                                batch_size=self.batch_size, 
                                shuffle = True, 
                                num_workers =self.num_workers, 
                                drop_last = True, 
                                pin_memory= self.pin_memory,
                                persistent_workers= self.persist_workers
                                ) #! persistent_workers=True: worker를 메모리에 유지
        
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
                        use_simple_rmsnorm = self.config['model']['use_simple_rmsnorm'],
                        cls_init=self.config['model']['cls_init'],
                        dropout_mask = self.config['model']['dropout_mask']
        )
        optimizer = AdamW(model.parameters(), lr = self.lr)
        num_training_steps = self.epochs * len(train_loader)  # 전체 학습에 걸쳐서 수행되는 step 수
        
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps= self.config['train']['warmup_steps'], 
                                                    num_training_steps=num_training_steps)
            
        criterion_ins = InstanceLoss(tau = self.config['model']['instance_tau'])
        criterion_clu = ClusterLoss(tau = self.config['model']['cluster_tau'])

        # WandB 로깅 설정
        if self.config['train']['use_wandb']:
            wandb_config = {**self.config['train'], **self.config['model']}

        device = self.device
        model, criterion_ins, criterion_clu, train_loader, optimizer, scheduler = \
            model.to(device), criterion_ins.to(device), criterion_clu.to(device), train_loader, optimizer, scheduler
        
        if self.config['gpu'] == 'multi':
            model = nn.DataParallel(model)
            device = f"cuda:{torch.cuda.current_device()}"
        else:
            print("single GPU")
            device = self.device

        # Finally train
        best_loss = 1e10
        best_model_dict = None
        training_step = 0
        
        epochs = self.epochs
        max_train_steps = epochs * len(train_loader)
        progress_bar = tqdm(range(max_train_steps), position=1)
        
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
                None,
                False,
                self.config['train']['use_wandb']
            )
            
            # Log
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")
        
          
        self.model = model
        state_dict = model.state_dict()
        clear_gpu_memory()
        
        # model saving
        # if not os.path.exists(f"{self.saving_path}/batch_{self.batch_size}_n_bins_{self.n_bins}_hidden_{self.hidden_dim}_std_{self.std}_mask_{self.mask}_ctau_{self.cluster_tau}/models"):
        #     os.makedirs(f"{self.saving_path}/batch_{self.batch_size}_n_bins_{self.n_bins}_hidden_{self.hidden_dim}_std_{self.std}_mask_{self.mask}_ctau_{self.cluster_tau}/models")
        # torch.save(state_dict, f"{self.saving_path}/batch_{self.batch_size}_n_bins_{self.n_bins}_hidden_{self.hidden_dim}_std_{self.std}_mask_{self.mask}_ctau_{self.cluster_tau}/models/{file_path.split('/')[-1].split('.')[0]}.pt")

        if self.config['train']['use_wandb']:
            wandb.finish()

        return model
    def cluster_inference(self, file_path, config, model):
         #* 동일한 데이터를 사용한다.
        test_data_frame = pd.read_csv(file_path)

        processed_data = preprocess_data(test_data_frame, file_path, logger)

        # Load data
        test_data = Embedding_dataset(self.config, data=processed_data)
        test_loader = DataLoader(test_data, 
                                batch_size=self.batch_size, 
                                shuffle=False, 
                                num_workers =self.num_workers, 
                                drop_last =False, 
                                pin_memory= self.pin_memory,
                                persistent_workers= self.persist_workers
                                ) #! persistent_workers=True: worker를 메모리에 유지
        
        # self.model.load_state_dict(torch.load(f"{self.saving_path}/best_model_{self.config['model']['cluster_num']}_{file_path}.pt"))
        # self.model.load_state_dict(torch.load(f"{self.saving_path}/{self.config['model']['cluster_num']}_{file_path}.pt"))
        if config['gpu'] == 'single':
            features, prob, clusters = inference(test_loader, self.model, self.device)
        else:
            features, prob, clusters = inference_parallel(test_loader, self.model, self.device)

        # No outlier
        clusters = clusters + 1
        predictions = pd.DataFrame({'firms': test_data_frame['PERMNO'], 'clusters': clusters, 'MOM1': test_data_frame['MOM1']})

        batch = config['train']['batch_size']
        n_bins = config['model']['n_bins']
        hidden_dim = config['model']['hidden_dim']
        std = config['model']['augment_std']
        mask = config['model']['masking_ratio']

        ##! save
        # Create base paths
        base_pred_path = os.path.join(self.saving_path, f"batch_{batch}_n_bins_{n_bins}_hidden_{hidden_dim}_std_{std}_mask_{mask}_ctau_{self.cluster_tau}", "predictions")
        base_prob_path = os.path.join(self.saving_path, f"batch_{batch}_n_bins_{n_bins}_hidden_{hidden_dim}_std_{std}_mask_{mask}_ctau_{self.cluster_tau}", "prob")
        
        # Get filename without path
        filename = os.path.basename(file_path)
        
        # Create prediction directory and save predictions
        pred_dir = os.path.join(base_pred_path, str(self.config['model']['cluster_num']))
        os.makedirs(pred_dir, exist_ok=True)
        pred_file = os.path.join(pred_dir, filename)
        predictions.to_csv(pred_file, index=False)

        #* probability saving
        prob_df = pd.DataFrame(prob, columns=[f'prob_{i}' for i in range(self.config['model']['cluster_num'])])
        prob_df['firms'] = test_data_frame['PERMNO']

        # Create probability directory and save probabilities 
        prob_dir = os.path.join(base_prob_path, str(self.config['model']['cluster_num']))
        os.makedirs(prob_dir, exist_ok=True)
        prob_file = os.path.join(prob_dir, filename)
        prob_df.to_csv(prob_file, index=False)

# ! Single GPU
def main_single(config):
    csv_loader = Get_Data(config['data'], config['data_refine'])
    file_list = csv_loader.get_file_list()
    for file in tqdm(file_list, desc="Processing files", unit="file"):
        trainer = ModelTrainer(config)
        start_time = datetime.now()
        model = trainer.train(file)
        trainer.cluster_inference(file, config, model)
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        tqdm.write(f"Processed {file} in {elapsed_time}")

if __name__ == "__main__":

    args = load_args()
    config = load_yaml_param_settings(args.config)
    
    # config 전체 logging
    name_of_run = f'{config["model"]["cluster_num"]}_{config["train"]["batch_size"]}_{config["model"]["n_bins"]}_{config["model"]["hidden_dim"]}_{config["model"]["augment_std"]}_{config["model"]["masking_ratio"]}_{config["model"]["cluster_tau"]}'
    logger = get_logger(name_of_run)
    logger.info("\n" + yaml.dump(config, indent=4, default_flow_style=False))

    if args.overwrite:
        config['model']['cluster_num'] = args.cluster_num
        config['train']['batch_size'] = args.batch_size
        logger.info("Overwrite the existing configuration")

    logger.info(f"\n Number of Cluster: {config['model']['cluster_num']}, Batch size: {config['train']['batch_size']}, Data_refine path: {config['data_refine']},\
            \n n_bins: {config['model']['n_bins']}, hidden_dim: {config['model']['hidden_dim']}, masking_ratio: {config['model']['masking_ratio']}, augment_std: {config['model']['augment_std']}, cluster_tau: {config['model']['cluster_tau']}")

    # Save the configuration
    if not os.path.exists(config['train']['saving_path']):
        os.makedirs(config['train']['saving_path'])
    date = datetime.now().strftime("%Y%m%d-%H%M")
    with open(f"{config['train']['saving_path']}/RunConfigs/run_{date}_config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    main_single(config)