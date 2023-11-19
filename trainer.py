
import time
import numpy as np
import torch
import util
import torch.nn as nn

from torch.utils.data import DataLoader


def train(model, train_ds, test_ds, loss_fn, \
          args, cost_norm=None, optimizer=None, scheduler=None):
    to_pred, bs, device, epochs, clip_size = \
        args.to_predict, args.bs, args.device, args.epochs, args.clip_size
    lr = args.lr

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if not scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.7)
    
    t0 = time.time()

    # best_prev = 999999
    total_train_step = 0
    train_dataloader = DataLoader(dataset=train_ds, batch_size=bs, shuffle=True)
    # speed up
    model.to(device)
    loss_fn.to(device) 

    for epoch in range(epochs):
        losses = 0
        cost_predss = np.empty(0)

        model.train()

        for data in train_dataloader:
            # print(data)
            optimizer.zero_grad()
            batch, batch_labels = data
            # speed up
            batch.to(device)
            batch_labels.to(device)

            outputs = model(batch)
            loss = loss_fn(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_size)
            optimizer.step()

            # 打印相关信息
            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                print("训练次数: {}, Loss: {}".format(total_train_step, loss.item()))
        
        if epoch > 40:
            # test_scores, corrs = evaluate(model, val)
            pass

        if epoch % 20 == 0:
            pass

        scheduler.step()

# def evaluate(model, ds, crit, bs, norm, device, prints=False):
#     model.eval()
#     cost_predss = np.empty(0)
#     train_dataloader = DataLoader(dataset=ds, batch_size=bs, shuffle=True) #???
#     total_test_loss = 0
#     with torch.no_grad():
#         for data in DataLoader:
#             batch, batch_labels = data

#             outputs = model(batch)

#             loss = crit(outputs, batch_labels)
#             total_test_loss = total_test_loss + loss.item()

#             # TODO: ?????
