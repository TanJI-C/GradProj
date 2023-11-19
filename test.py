import pandas as pd
import sys
import json
import torch
from torch.utils.data import DataLoader

import util as tu
import model as tm
import trainer as tt

imdb_path = './data/imdb/'
full_train_df = pd.DataFrame()
# for i in range(1):
file = imdb_path + 'plan_and_cost/train_plan_part1.csv'
df = pd.read_csv(file)
full_train_df = pd.concat([full_train_df, df], ignore_index=True)
dict_list = tu.Encoder.format_imdb(full_train_df)

train_dataset = tu.PlanTreeDataSet(dict_list)
# 
full_test_df = pd.DataFrame()
file = imdb_path + 'plan_and_cost/test_plan_part.csv'
df = pd.read_csv(file)
full_test_df = pd.concat([full_test_df, df], ignore_index=True)
dict_list = tu.Encoder.format_imdb(full_test_df)

test_dataset = tu.PlanTreeDataSet(dict_list)



model = tm.PlanTreeModel(seq_type_dim=15, seq_hidden_dim=512, seq_output_dim=16, output_dim=1)


class Args:
    bs = 1000
    lr = 0.001
    epochs = 200
    clip_size = 50
    embed_size = 64
    pred_hid = 128
    ffn_dim = 128
    head_size = 12
    n_layers = 8
    dropout = 0.1
    sch_decay = 0.6
    device = 'cpu:0'
    newpath = './results/full/cost/'
    to_predict = 'cost'
args = Args()


class LogMSELoss(torch.nn.Module):
    def __init__(self):
        super(LogMSELoss, self).__init__()

    def forward(self, input, target):
        mse_loss = torch.nn.functional.mse_loss(input, target)
        log_mse_loss = torch.log(mse_loss)
        return log_mse_loss


tt.train(model, train_dataset, test_dataset, LogMSELoss(), args)
