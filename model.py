import torch
import torch.nn as nn
import util
from typing import Any, List, Optional, Sequence, Tuple

class SeqFormer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(SeqFormer, self).__init__()
        # input_dim: node bits
        self.tranformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim, dim_feedforward=hidden_dim, nhead=1, batch_first=True
            ),
            num_layers=1,
        )
        self.node_length = input_dim
        p = 0.2
        self.mlp = nn.Sequential(
            *[
                nn.Linear(self.node_length, 128),
                nn.Dropout(p),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.Dropout(p),
                nn.ReLU(),
                nn.Linear(64, output_dim),
            ]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # change x shape to (batch, seq_len, input_size) from (batch, len)
        # one node is 9 bits
        x = x.view(x.shape[0], -1, self.node_length) # 保证是三维的，第一维是batch，第二维是seq_len，第三维是input_size
        out = self.tranformer_encoder(x)
        out = self.mlp(out[:, 0, :]) # 只取第一个node的输出，即只预测根节点
        out = self.sigmoid(out)
        return out


    # # test case
    # input_data = torch.randn(512, 40, 9) # 即一共有512个plan，每个plan有40个node，每个node有9个特征（特征包括one-hot节点类型、cost、card）

    # model = SeqFormer(9, 512, 1) # 第一个是输入的特征维度，第二个是transformer的hidden维度（先用512就行），第三个是输出的维度

    # output = model(input_data) # 每个plan的预测结果

    # print(output.shape)

class PlanTreeModel(nn.Module):
    def __init__(self, seq_type_dim, seq_hidden_dim, seq_output_dim, output_dim) -> None:
        super(PlanTreeModel, self).__init__()
        self.seqformer = SeqFormer(input_dim=seq_type_dim, hidden_dim=seq_hidden_dim, output_dim=seq_output_dim)
        p = 0.2
        self.mlp = nn.Sequential(
            *[
                nn.Linear(seq_output_dim + 2, 128),
                nn.Dropout(p),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.Dropout(p),
                nn.ReLU(),
                nn.Linear(64, output_dim),
            ]
        )
    
    def forward(self, input: util.BatchData):
        type_data = input['type_feature']
        info_data = input['info_feature']
        
        output1 = self.seqformer(type_data) # 得到一个seq_output_dim维度的tensor,(batch, seq_output_dim)
        # 和info_data(batch, 2)合并, 得到(batch, seq_output_dim+2)的二维tensor，作为输入进行训练
        output2 = self.mlp(torch.cat((output1, info_data), dim=1))
        return output2

