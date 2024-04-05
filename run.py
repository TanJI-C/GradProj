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


configs = Namespace (
    project_name = "TanJI",
    test_database_ids = [13],
    random_seed = 114514,
    batch_size = 1000,
    accelerator = 'cpu',
    progress_bar = True,
    lr = 0.001,
    dropout = 0.2,
    epochs = 100,
    pad_length = 20,
    node_length = 19,
    hidden_dim = 192,
    output_dim = 1,
    newpath = './results/full/cost/',
    to_predict = 'cost',
    dataset_file_name = 'alldataset.pkl',
)

with open(path.join(ROOT_DIR, 'data', 'workload1', 'statistics.json')) as file:
    feature_statistics = json.load(file)
configs.node_length = len(feature_statistics['node_types']['value_dict']) + 2
# with open(path.join(ROOT_DIR, 'data', 'imdb', 'statistics.json')) as file:
#     feature_statistics = json.load(file)
encoder = tu.Encoder(feature_statistics, configs)


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

for wk_item in workloads:
    files_name.append(path.join(ROOT_DIR, 'data', 'workload1', wk_item + '_filted.json'))
# for i in range(20):
#     files_name.append(path.join(ROOT_DIR, 'data', 'imdb', 'plan_and_cost', 'train_plan_part{}.csv'.format(i)))

data_file_name = path.join(ROOT_DIR, 'data', 'workload1', configs.dataset_file_name)
# data_file_name = path.join(ROOT_DIR, 'data', 'imdb', configs.dataset_file_name)

if os.path.exists(data_file_name):
    alldataset = load_instance(data_file_name)
else:
    dict_list = tu.Encoder.format_workload(files_name)
    # dict_list = tu.Encoder.format_imdb(files_name)
    alldataset = tu.PlanTreeDataSet(dict_list, encoder)
    save_instance(alldataset, data_file_name)

train_dataloader, val_dataloader, test_dataloader = tu.get_dataloader(alldataset, configs)

nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
model = tm.TanJI(configs)
logger = pl.loggers.WandbLogger(project=configs.project_name, name=nowtime)
logger.log_hyperparams(configs.__dict__)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=os.path.join(ROOT_DIR, "checkpoints"),
    filename="DACE",
)
trainer = tm.PLTrainer(
    accelerator=configs.accelerator,
    enable_progress_bar=configs.progress_bar,
    enable_model_summary=configs.progress_bar,
    max_epochs=configs.epochs,
    logger=logger,
    callbacks=[checkpoint_callback],
)

# trainer = PLTrainer(accelerator="cpu", max_epochs=50, logger=wandb_logger)
trainer.fit(model, train_dataloader, val_dataloader)

# test model
result = trainer.test(model, dataloaders=test_dataloader)