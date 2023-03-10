from .AudioUtil import AudioUtil
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
import torch

# ----------------------------
# Sound Dataset，声音数据集
# ----------------------------
class SoundDS(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 5000
        self.sr = 48000
        self.channel = 2
        self.shift_pct = 0.3
    
    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)
    
    # ----------------------------
    # 从数据集中取第idx项，并且经过一番预处理
    # ----------------------------
    def __getitem__(self, idx):
        # 获取到数据集文件路径
        audio_file = self.df.loc[idx, 'path']
        # 获取到数据集的标签
        class_id = self.df.loc[idx, 'label']

        # audio_file = str('./train/fc802bee-ef01-4496-a898-d7e77c0017e9.wav')
        # class_id = 0

        audio_file = 'G:/深度学习/小项目/新冠检测/Classifier-Model/'+audio_file
        # print('sound: ', audio_file)
        aud = AudioUtil.open(audio_file)

        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)

        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        # 生成梅尔谱图
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        # 将生成的梅尔谱图进行优化，提高泛化能力
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        # 返回的是经过处理后的第idx项数据、其标签
        return aug_sgram, class_id


class InitialSoundDS(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 5000
        self.sr = 48000
        self.channel = 2
        self.shift_pct = 0.3
    
    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)
    
    # ----------------------------
    # 从数据集中取第idx项，并且经过一番预处理
    # ----------------------------
    def __getitem__(self, idx):
        # 获取到数据集文件路径
        audio_file = self.df.loc[idx, 'path']
        # 获取到数据集的标签
        class_id = self.df.loc[idx, 'label']

        audio_file = 'G:/深度学习/小项目/新冠检测/Classifier-Model/'+audio_file
        # print('sound: ', audio_file)
        data, sr = torchaudio.load(audio_file)
        # 返回的是经过处理后的第idx项数据、其标签
        return data.numpy().tolist(), class_id