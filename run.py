from os import path
import pandas as pd
import sys
import json
import torch
from torch.utils.data import DataLoader

import util as tu
import model as tm
import trainer as tt
from setup import *

files_name = []

for wk_item in workloads:
    files_name.append(path.join(ROOT_DIR, 'data', 'workload1', wk_item))
print(files_name)