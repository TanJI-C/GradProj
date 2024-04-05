from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import util
import lightning.pytorch as pl
from typing import Any, List, Optional, Sequence, Tuple


class TanJI(pl.LightningModule):
    def __init__(self, configs):
        super(TanJI, self).__init__()
        # input_dim: node bits
        self.tranformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=configs.node_length, dim_feedforward=configs.hidden_dim, nhead=1, batch_first=True,
                activation="gelu",
                dropout=configs.dropout
            ),
            num_layers=1,
        )
        self.node_length = configs.node_length
        self.mlp_dim = [configs.node_length * configs.pad_length, 128, 64]
        self.mlp = nn.Sequential(
            *[
                nn.Linear(self.mlp_dim[0], self.mlp_dim[1]),
                nn.Dropout(configs.dropout),
                nn.ReLU(),
                nn.Linear(self.mlp_dim[1], self.mlp_dim[2]),
                nn.Dropout(configs.dropout),
                nn.ReLU(),
                nn.Linear(self.mlp_dim[2], configs.output_dim),
            ]
        )
        self.sigmoid = nn.Sigmoid()

        self.configs = configs

    def forward(self, feature, atten_mask):
        # change x shape to (batch, seq_len, input_size) from (batch, len)
        # one node is 9 bits
        feature = feature.view(feature.shape[0], -1, self.node_length) # 保证是三维的，第一维是batch，第二维是seq_len，第三维是input_size
        out = self.tranformer_encoder(feature, mask=atten_mask)
        out = out.view(feature.shape[0], -1)
        out = self.mlp(out)
        out = self.sigmoid(out)
        return out
    
    def loss_q_error(self, input, target, loss_mask):
        loss = torch.max(input / target, target / input)
        loss = loss * loss_mask
        loss = torch.log(torch.where(loss > 1, loss, 1))
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss)
        return loss
    
    def training_step(self, batch, batch_idx):
        seqs, atten_mask, loss_mask, labels = batch
        outputs = self(seqs, atten_mask)
        # loss = self.loss_q_error(outputs, labels, loss_mask)
        loss = util.q_error(outputs[:, 0], labels[:, 0])
        loss = torch.mean(loss)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        seqs, atten_mask, loss_mask, labels = batch
        outputs = self(seqs, atten_mask)
        loss = util.q_error(outputs[:,0], labels[:,0])
        loss = torch.mean(loss)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        seqs, atten_mask, loss_mask, labels = batch
        outputs = self(seqs, atten_mask)
        loss = self.loss_q_error(outputs[:,0], labels[:,0], loss_mask)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.configs.lr)
        return optimizer

# create pytorch_lightning trainer and overwrite the test function
class PLTrainer(pl.Trainer):
    # succeed the init function
    def __init__(self, *args, **kwargs):
        super(PLTrainer, self).__init__(*args, **kwargs)

    def test(self, model, dataloaders=None, ckpt_path=None, test_data_name=None):
        if dataloaders is None:
            if self.test_dataloaders is None:
                raise ValueError(
                    "Trainer that returned None for test_dataloaders or passed None to test"
                )
            dataloaders = self.test_dataloaders

        model.eval()

        # get q-error of all test data
        qerrors = []
        for seqs, attn_masks, loss_mask, labels in dataloaders:
            outputs = model(seqs, attn_masks)
            qerror = util.q_error(outputs[:,0], labels[:,0])
            qerrors.append(qerror)
        qerrors = torch.cat(qerrors, dim=0)
        # save test loss, median, 90th, 95th, 99th, max and mean in a dict
        test_metrics = {}
        # test_metrics["50th test loss"] = torch.quantile(qerrors, 0.5).item()
        # test_metrics["90th test loss"] = torch.quantile(qerrors, 0.9).item()
        # test_metrics["95th test loss"] = torch.quantile(qerrors, 0.95).item()
        # test_metrics["99th test loss"] = torch.quantile(qerrors, 0.99).item()
        # test_metrics["mean test loss"] = torch.mean(qerrors).item()
        # test_metrics["max test loss"] = torch.max(qerrors).item()

        test_metrics[test_data_name + " 50th test loss"] = torch.quantile(qerrors, 0.5).item()


        # report test loss, median, 90th, 95th, 99th, max and mean in logger
        for k, v in test_metrics.items():
            self.logger.log_metrics({k: v})
        return test_metrics
