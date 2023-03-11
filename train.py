from torch.utils.data import random_split
import pandas as pd
from pathlib import Path
import torch
from torch import nn
import torchaudio
from model.resNet import ResNet
from model.complexNet import ComplexNet

# ----------------------------
# Training Loop
# ----------------------------
def training(model: nn.Module, train_dl, num_epochs, logger, device):
    logger.info('Start training!')
    # 损失函数使用交叉熵
    criterion = nn.CrossEntropyLoss()
    # 优化器使用Adam
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')
    # Repeat for each epoch，开始迭代训练
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        for i, data in enumerate(train_dl):
            # inputs, labels = data[0].to(device), data[1].to(device)
            inputs: torch.Tensor
            labels: torch.Tensor 
            inputs, labels = data[0].to(device), data[1].to(device)
            # print(inputs.shape, inputs) # inputs: (16, 2, h, w)
            
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            optimizer.zero_grad()
            # 经过模型计算出来的结果，是一个二维向量
            outputs = model(inputs)
            # print(type(outputs), type(labels))
            loss = criterion(outputs, labels.long()) # 损失函数定义中，target必须为long
            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            _, prediction = torch.max(outputs,1)

            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            # if i % 10 == 0:
                # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
    
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        # print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
        logger.info('Epoch:[{}/{}]\t average loss={:.5f}\t total loss={:.5f}\t acc={:.3f}'.format(epoch+1 , num_epochs, avg_loss, running_loss, acc ))
        
    logger.info('Finished Training')