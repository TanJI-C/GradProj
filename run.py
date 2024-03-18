from os import path
import pandas as pd
import sys
import json
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import os
import wandb
from argparse import Namespace
import datetime

import util as tu
import model as tm
from setup import *


config = Namespace (
    project_name = "GradProj",

    test_database_ids = [13],
    random_seed = 114514,
    batch_size = 512,
    accelerator = 'cpu',
    progress_bar = True,
    lr = 0.001,
    max_epoch = 50,
    clip_size = 50,
    embed_size = 64,
    pred_hid = 128,
    ffn_dim = 128,
    head_size = 12,
    n_layers = 8,
    dropout = 0.1,
    sch_decay = 0.6,
    device = 'cpu:0',
    newpath = './results/full/cost/',
    to_predict = 'cost',

    dataset_file_name = 'alldataset.pkl'
)

with open(path.join(ROOT_DIR, 'data', 'workload1', 'statistics.json')) as file:
    feature_statistics = json.load(file)
encoder = tu.Encoder(feature_statistics)

import pickle

# 存储类的实例
def save_instance(instance, filename):
    with open(filename, 'wb') as file:
        pickle.dump(instance, file)

# 加载存储的实例
def load_instance(filename):
    with open(filename, 'rb') as file:
        instance = pickle.load(file)
    return instance



# for zero-shot data
files_name = []

# for wk_item in workloads:
#     files_name.append(path.join(ROOT_DIR, 'data', 'workload1', wk_item + '_filted.json'))
for i in range(18):
    files_name.append(path.join(ROOT_DIR, 'data', 'imdb', 'plan_and_cost', 'train_plan_part{}.csv'.format(i)))

# data_file_name = path.join(ROOT_DIR, 'data', 'workload1', config.dataset_file_name)
data_file_name = path.join(ROOT_DIR, 'data', 'imdb', config.dataset_file_name)

if os.path.exists(data_file_name):
    alldataset = load_instance(data_file_name)
else:
    # dict_list = tu.Encoder.format_workload(files_name)
    dict_list = tu.Encoder.format_imdb(files_name)
    alldataset = tu.PlanTreeDataSet(dict_list, encoder)
    save_instance(alldataset, data_file_name)

train_dataloader, val_dataloader, test_dataloader = tu.get_dataloader(alldataset, config)


nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
model = tm.SeqFormer(config, node_length=18, pad_length=20, hidden_dim=192, output_dim=1)
logger = pl.loggers.WandbLogger(project=config.project_name, name=nowtime)
logger.log_hyperparams(config.__dict__)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=os.path.join(ROOT_DIR, "checkpoints"),
    filename="DACE",
)
trainer = tm.PLTrainer(
    accelerator=config.accelerator,
    enable_progress_bar=config.progress_bar,
    enable_model_summary=config.progress_bar,
    max_epochs=config.max_epoch,
    logger=logger,
    callbacks=[checkpoint_callback],
)

# trainer = PLTrainer(accelerator="cpu", max_epochs=50, logger=wandb_logger)
trainer.fit(model, train_dataloader, val_dataloader)


# test model
result = trainer.test(model, dataloaders=test_dataloader)