from torch.utils.data import random_split
from utils.SoundDS import SoundDS, InitialSoundDS
from utils.collate import collate_fn
from model.resNet import ResNet
from model.complexNet import ComplexNet
from train import training
from inference import inference
import pandas as pd
import torch

if __name__ == '__main__':
    # 读取训练集文件路径
    metadata_file = 'G:\深度学习\小项目\新冠检测\Classifier-Model\\train.csv'
    data_path = './data/train/'
    df = pd.read_csv(metadata_file)

    # myds = SoundDS(df, data_path)
    myds = InitialSoundDS(df, data_path)

    num_items = len(myds)
    # print("number: ", num_items)
    # 训练集和验证集以4:1进行分割
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    # 将myds数据集随机划分为训练集和验证集
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    # 得到训练集和验证集的数据加载器
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    # model = ResNet()
    model = ComplexNet()
    myModel = model.to(device)
    
    # 训练
    training(model=model, train_dl=train_dl, num_epochs=10, device=device)

    # 测试
    inference(model, val_dl, device=device)