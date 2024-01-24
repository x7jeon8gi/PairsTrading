from argparse import ArgumentParser
import os
from pathlib import Path
import yaml
import pickle
import logging
import torch


def get_root_dir():
    return Path(__file__).parent.parent

def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    return parser.parse_args()


def load_yaml_param_settings(yaml_fname: str):
    """
    :param yaml_fname: .yaml file that consists of hyper-parameter settings.
    """
    stream = open(yaml_fname, 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def load_args_notebook():
    """For Jupyter Notebook"""
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=Path('').joinpath('./configs', 'config.yaml'))
    
    #* for jupyter notebook
    return parser.parse_args(args=[])

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    return model