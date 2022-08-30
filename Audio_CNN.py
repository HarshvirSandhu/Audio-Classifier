# import tarfile
# file = tarfile.open('C:/Users/harsh/Downloads/UrbanSound8K.tar.gz')
# file.extractall('C:/Users/harsh/PycharmProjects/Harshvir_S/Audio_pytorch')

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio
import torchaudio.transforms as transforms
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.models import resnet18


# torchaudio.load :=> Returns signal tensor and sampling rate
# print(torchaudio.load('C:/Users/harsh/PycharmProjects/Harshvir_S/Audio_pytorch/UrbanSound8K/audio/fold5/100032-3-0-0.wav'))
# signal, sr = torchaudio.load('C:/Users/harsh/PycharmProjects/Harshvir_S/Audio_pytorch/UrbanSound8K/audio/fold5/100032-3-0-0.wav')


class Urban_Sound_8K(Dataset):
    def __init__(self, annotations, audio_dir, mel_spectrogram, new_sr, num_samples):
        self.annotations = annotations
        self.audio_dir = audio_dir
        self.df = pd.read_csv(annotations)
        self.mel_spectrogram = mel_spectrogram
        self.new_sr = new_sr  # Sampling rate that all signals will have
        self.num_samples = num_samples

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        label = self.df['classID'][item]
        fold_num = self.df['fold'][item]
        fold = f'fold{fold_num}'
        name = self.df['slice_file_name'][item]
        path = os.path.join(self.audio_dir, fold, name)
        signal, sr = torchaudio.load(path)

        # Converting signal to mel spectrogram
        signal = self.signal_mix_down(signal)
        signal = self.resample_fn(signal, sr)
        signal = self.crop_signal(signal)
        signal = self.pad_signal(signal)
        signal = self.mel_spectrogram(signal)
        return signal, label

    def signal_mix_down(self, signal):  # Converting signal of multiple channel to a single channel
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def resample_fn(self, signal, sr):  # Resampling so that all signals have the same sampling rate.
        if sr != self.new_sr:
            resample = transforms.Resample(sr, new_freq=self.new_sr)
            signal = resample(signal)
        return signal

    def crop_signal(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :num_samples]
        return signal

    def pad_signal(self, signal):
        len_pad = self.num_samples - signal.shape[1]
        if len_pad > 0:
            pad = (0, len_pad)
            signal = F.pad(signal, pad)
        return signal


labels = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

sample_rate = 22000
num_samples = 22000

mel_spectrogram = transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=64)
usd = Urban_Sound_8K(
    annotations='C:/Users/harsh/PycharmProjects/Harshvir_S/Audio_pytorch/UrbanSound8K/metadata/UrbanSound8K.csv',
    audio_dir='C:/Users/harsh/PycharmProjects/Harshvir_S/Audio_pytorch/UrbanSound8K/audio',
    mel_spectrogram=mel_spectrogram,
    new_sr=sample_rate,
    num_samples=num_samples
)
print(usd.__len__())
print(usd.__getitem__(0)[0].shape, usd.__getitem__(0)[1])
# save_image(usd.__getitem__(0)[0], 'mel_spectrogram.jpg')

batch_size = 64
lr = 3e-4
num_epochs = 10
model = resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(512, 10)
# print(model)
data_load = DataLoader(dataset=usd, batch_size=batch_size, shuffle=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# epoch_loss = 0
print(len(data_load))

# for epoch in range(1):
#     epoch_loss = 0
#     for i, (data, label) in enumerate(data_load):
#         if i % 50 == 0 and i >= 50:
#             print(f'{i}/{len(data_load)}')
#             print(data.shape)
#         print(i[0].shape, i[1].shape)
        # optimizer.zero_grad()
        # data = data.repeat(1, 3, 1, 1)
        # score = model(data)
        # loss = criterion(score, label)
        # print(loss)
        # loss.backward()
        # optimizer.step()
        # epoch_loss += loss
    # print(epoch_loss)

signal, sr = torchaudio.load(
    'C:/Users/harsh/PycharmProjects/Harshvir_S/Audio_pytorch/UrbanSound8K/audio/fold5/100032-3-0-0.wav')

new_sr = sample_rate


def signal_mix_down(signal):  # Converting signal of multiple channel to a single channel
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal


def resample_fn(signal, sr):  # Resampling so that all signals have the same sampling rate.
    if sr != new_sr:
        resample = transforms.Resample(sr, new_freq=new_sr)
        signal = resample(signal)
    return signal


def crop_signal(signal):
    if signal.shape[1] > num_samples:
        signal = signal[:, :num_samples]
    return signal


def pad_signal(signal):
    len_pad = num_samples - signal.shape[1]
    if len_pad > 0:
        pad = (0, len_pad)
        signal = F.pad(signal, pad)
    return signal


signal = signal_mix_down(signal)
signal = resample_fn(signal, sr)
signal = crop_signal(signal)
signal = pad_signal(signal)
signal = mel_spectrogram(signal)
print(signal.unsqueeze(0).repeat(1, 3, 1, 1).shape)
print(F.softmax(model(signal.unsqueeze(0).repeat(1, 3, 1, 1))))

# signal, sr = torchaudio.load(
#     'C:/Users/harsh/PycharmProjects/Harshvir_S/few_shot_learning_project/truth_lie_dataset/Train/Truth/Recording(59).m4a')

# print(sr, signal.shape)
record_dir = 'C:/Users/harsh/PycharmProjects/Harshvir_S/few_shot_learning_project/truth_lie_dataset/Train/Truth'

from pydub import AudioSegment
for file_path in os.listdir(record_dir):
    file_path = record_dir+'/'+file_path
    print(file_path)
    audio = AudioSegment.from_file(file_path, format='m4a')
    dest_path = file_path.split(sep='.')[0] + '.wav'
    print(dest_path)
    audio.export(dest_path, format='wav')
    break