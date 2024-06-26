"""
This is the integrated dataset collection. 
Collate functions should be explicitly included in the definition of dataset as static method, 
instead independently defined outside so as to avoid overriding. 

"""
import pandas as pd
from torch.utils.data import Dataset
import torch
import torchaudio
import os
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from misc_my_utils import time_to_rel_frame
import torch.nn.functional as F

class SeqDataset(Dataset):
    """
    A PyTorch dataset that loads cutted wave files from disk and returns input-output pairs for
    training autoencoder. [wav -> mel]
    """
    
    def __init__(self, load_dir, load_control_path, transform=None):
        """
        Initializes the class by reading a CSV file and merging the "rec" and "idx" columns.

        Args:
        load_dir (str): The directory containing the files to load.
        load_control_path (str): The path to the CSV file containing the "rec" and "idx" columns.
        transform (Transform): when loading files, this will be applied to the sound data. 
        """
        control_file = pd.read_csv(load_control_path)
        control_file = control_file[control_file['n_frames'] > 400]
        control_file = control_file[control_file['duration'] <= 2.0]
        
        # Extract the "rec" and "idx" columns
        rec_col = control_file['rec'].astype(str)
        idx_col = control_file['idx'].astype(str).str.zfill(8)
        
        # Merge the two columns by concatenating the strings with '_' and append extension name
        merged_col = rec_col + '_' + idx_col + ".wav"
        
        self.dataset = merged_col.tolist()
        self.load_dir = load_dir
        self.transform = transform
        
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Returns a tuple (input_data, output_data) for the given index.

        The function first checks if the provided index is a tensor, and if so, converts it to a list.
        It then constructs the file path for the .wav file using the dataset attribute and the provided index.
        The .wav file is loaded using torchaudio, and its data is normalized. If a transform is provided,
        the data is transformed using the specified transform. Finally, the input_data and output_data are
        set to the same data (creating a tuple), and the tuple is returned.

        Note: 
        This function assumes that the class has the following attributes:
        - self.load_dir (str): The directory containing the .wav files.
        - self.dataset (list): A list of .wav file names.
        - self.transform (callable, optional): An optional transform to apply to the audio data.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        wav_name = os.path.join(self.load_dir,
                                self.dataset[idx])
        
        data, sample_rate = torchaudio.load(wav_name, normalize=True)
        if self.transform:
            data = self.transform(data)
        
        return data

    @staticmethod
    def collate_fn(xx):
        # only working for one data at the moment
        batch_first = True
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return xx_pad, x_lens



class SeqDatasetInfo(Dataset):
    def __init__(self, load_dir, load_control_path, transform=None):
        control_file = pd.read_csv(load_control_path)
        control_file = control_file[control_file['n_frames'] > 400]
        control_file = control_file[control_file['duration'] <= 2.0]
        
        # Extract the "rec" and "idx" columns
        rec_col = control_file['rec'].astype(str)
        idx_col = control_file['idx'].astype(str).str.zfill(8)

        # Extract the "token" and "produced_segments" columns
        token_col = control_file['token'].astype(str)
        produced_segments_col = control_file['produced_segments'].astype(str)
        
        # Merge the two columns by concatenating the strings with '_' and append extension name
        merged_col = rec_col + '_' + idx_col + ".wav"
        
        self.dataset = merged_col.tolist()
        self.infoset = produced_segments_col.tolist()
        self.info_rec_set = rec_col.tolist()
        self.info_idx_set = idx_col.tolist()
        self.info_token_set = token_col.tolist()

        self.load_dir = load_dir
        self.transform = transform
        
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        wav_name = os.path.join(self.load_dir,
                                self.dataset[idx])
        
        data, sample_rate = torchaudio.load(wav_name, normalize=True)
        if self.transform:
            data = self.transform(data)
        
        info = self.infoset[idx]
        # extra info for completing a csv
        info_rec = self.info_rec_set[idx]
        info_idx = self.info_idx_set[idx]
        info_token = self.info_token_set[idx]
        
        return data, info, info_rec, info_idx, info_token

    @staticmethod
    def collate_fn(data):
        # xx = data, aa bb cc = info_rec, info_idx, info_token
        xx, yy, aa, bb, cc = zip(*data)
        # only working for one data at the moment
        batch_first = True
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return xx_pad, x_lens, yy, aa, bb, cc



class SeqDatasetAnno(Dataset):
    """
    SeqDataset with paired annotation
    """
    def __init__(self, load_dir, load_control_path, transform=None):
        """
        load_dir: dir of audio
        load_control_path: path to corresponding log.csv
        transform: mel / mfcc
        """
        control_file = pd.read_csv(load_control_path)
        control_file = control_file[control_file['n_frames'] > 400] # if <= 400, cannot make one frame, will cause error
        control_file = control_file[control_file['duration'] <= 2.0]
        
        # Extract the "rec" and "idx" columns
        rec_col = control_file['rec'].astype(str)
        idx_col = control_file['idx'].astype(str).str.zfill(8)

        # Extract the "token" and "produced_segments" columns
        token_col = control_file['token'].astype(str)
        
        # Merge the two columns by concatenating the strings with '_' and append extension name
        merged_col = rec_col + '_' + idx_col + ".wav"
        name_col = rec_col + '_' + idx_col
        
        self.dataset = merged_col.tolist()
        self.token_set = token_col.tolist()
        self.name_set = name_col.tolist()

        self.load_dir = load_dir
        self.transform = transform
        
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        wav_name = os.path.join(self.load_dir,
                                self.dataset[idx])
        
        data, sample_rate = torchaudio.load(wav_name, normalize=True)
        if self.transform:
            data = self.transform(data)

        # extra info for completing a csv
        token = self.token_set[idx]
        name = self.name_set[idx]
        
        return data, token, name
    
    @staticmethod
    def collate_fn(data):
        # xx = data, aa bb cc = info_rec, info_idx, info_token
        xx, token, name = zip(*data)
        # only working for one data at the moment
        batch_first = True
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return xx_pad, x_lens, token, name

class SeqDatasetBoundary(Dataset):
    """
    SeqDataset with paired boundary information
    """
    def __init__(self, load_dir, load_control_path, transform=None):
        """
        load_dir: dir of audio
        load_control_path: path to corresponding log.csv
        transform: mel / mfcc
        """
        control_file = pd.read_csv(load_control_path)
        control_file = control_file[control_file['n_frames'] > 400] # if <= 400, cannot make one frame, will cause error
        control_file = control_file[control_file['duration'] <= 2.0]
        # control_file = control_file[control_file['match_status'] == 1]  # if individual phoneme time range matches with word time range
        
        # Extract the "rec" and "idx" columns
        rec_col = control_file['rec'].astype(str)
        idx_col = control_file['idx'].astype(str).str.zfill(8)

        # t1 t2 ... tn -> ["t1", "t2", ..., "tn"] -> [t1, t2, ..., tn] -> [f1, f2, ..., fn]
        phoneme_boundaries_col = control_file.apply(time_to_rel_frame, axis=1)

        match_status_col = control_file['match_status'].astype(int)
        
        # Merge the two columns by concatenating the strings with '_' and append extension name
        merged_col = rec_col + '_' + idx_col + ".wav"
        name_col = rec_col + '_' + idx_col
        
        self.dataset = merged_col.tolist()
        self.bnd_set = phoneme_boundaries_col.tolist()
        self.name_set = name_col.tolist()
        self.ms_set = match_status_col.tolist()

        self.load_dir = load_dir
        self.transform = transform
        
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        wav_name = os.path.join(self.load_dir,
                                self.dataset[idx])
        
        data, sample_rate = torchaudio.load(wav_name, normalize=True)
        if self.transform:
            data = self.transform(data)

        # extra info for completing a csv
        bnd = self.bnd_set[idx]
        name = self.name_set[idx]
        match_status = self.ms_set[idx]
        
        return data, bnd, name, match_status
    
    @staticmethod
    def collate_fn(data):
        # xx = data, aa bb cc = info_rec, info_idx, info_token
        xx, bnd, name, match_status = zip(*data)
        # only working for one data at the moment
        batch_first = True
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return xx_pad, x_lens, bnd, name, match_status


class SeqDatasetName(Dataset):
    """
    A PyTorch dataset that loads cutted wave files from disk and returns input-output pairs for
    training autoencoder. [wav -> mel]
    """
    
    def __init__(self, load_dir, load_control_path, transform=None):
        """
        Initializes the class by reading a CSV file and merging the "rec" and "idx" columns.

        Args:
        load_dir (str): The directory containing the files to load.
        load_control_path (str): The path to the CSV file containing the "rec" and "idx" columns.
        transform (Transform): when loading files, this will be applied to the sound data. 
        """
        control_file = pd.read_csv(load_control_path)
        control_file = control_file[control_file['n_frames'] > 400]
        control_file = control_file[control_file['duration'] <= 2.0]
        
        # Extract the "rec" and "idx" columns
        rec_col = control_file['rec'].astype(str)
        idx_col = control_file['idx'].astype(str).str.zfill(8)
        
        # Merge the two columns by concatenating the strings with '_' and append extension name
        merged_col = rec_col + '_' + idx_col + ".wav"
        name_col = rec_col + '_' + idx_col
        
        self.dataset = merged_col.tolist()
        self.name_set = name_col.tolist()
        self.load_dir = load_dir
        self.transform = transform
        
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Returns a tuple (input_data, output_data) for the given index.

        The function first checks if the provided index is a tensor, and if so, converts it to a list.
        It then constructs the file path for the .wav file using the dataset attribute and the provided index.
        The .wav file is loaded using torchaudio, and its data is normalized. If a transform is provided,
        the data is transformed using the specified transform. Finally, the input_data and output_data are
        set to the same data (creating a tuple), and the tuple is returned.

        Note: 
        This function assumes that the class has the following attributes:
        - self.load_dir (str): The directory containing the .wav files.
        - self.dataset (list): A list of .wav file names.
        - self.transform (callable, optional): An optional transform to apply to the audio data.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        wav_name = os.path.join(self.load_dir,
                                self.dataset[idx])
        
        data, sample_rate = torchaudio.load(wav_name, normalize=True)
        if self.transform:
            data = self.transform(data)
        
        name = self.name_set[idx]
        
        return data, name

    @staticmethod
    def collate_fn(data):
        xx, name = zip(*data)
        # only working for one data at the moment
        batch_first = True
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return xx_pad, x_lens, name


class MelTransform(nn.Module):
    def __init__(self, sample_rate, n_fft=400, n_mels=64): 
        super().__init__()
        self.sample_rate = sample_rate
        n_stft = int((n_fft//2) + 1)
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=n_mels, n_fft=n_fft)
        self.inverse_mel = torchaudio.transforms.InverseMelScale(sample_rate=sample_rate, n_mels=n_mels, n_stft=n_stft)
        self.grifflim = torchaudio.transforms.GriffinLim(n_fft=n_fft)
    
    def forward(self, waveform):
        mel_spec = self.transform(waveform)
        mel_spec = mel_spec.squeeze()
        mel_spec = mel_spec.permute(1, 0) # (F, L) -> (L, F)
        return mel_spec
    
class MelSpecTransformNewNew(nn.Module): 
    def __init__(self, sample_rate, n_fft=400, n_mels=64, normalizer=None, denormalizer=None): 
        super().__init__()
        self.sample_rate = sample_rate
        n_stft = int((n_fft//2) + 1)
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=n_mels, n_fft=n_fft)
        self.inverse_mel = torchaudio.transforms.InverseMelScale(sample_rate=sample_rate, n_mels=n_mels, n_stft=n_stft)
        self.grifflim = torchaudio.transforms.GriffinLim(n_fft=n_fft)
        self.amplitude_to_DB = torchaudio.transforms.AmplitudeToDB(stype='power')

        self.normalizer = normalizer
        self.denormalizer = denormalizer

    def forward(self, waveform): 
        # transform to mel_spectrogram
        mel_spec = self.transform(waveform)  # (channel, n_mels, time)
        # mel_spec = F.amplitude_to_DB(mel_spec)
        mel_spec = self.amplitude_to_DB(mel_spec)
        # mel_spec = torch.tensor(librosa.power_to_db(mel_spec.squeeze().numpy()))
        mel_spec = mel_spec.squeeze()
        mel_spec = mel_spec.permute(1, 0) # (F, L) -> (L, F)
        mel_spec = self.normalizer(mel_spec)
        return mel_spec
    
    def de_norm(self, this_mel_spec, waveform): 
        # transform to mel_spectrogram
        mel_spec = self.transform(waveform)  # (channel, n_mels, time)
        # mel_spec = torch.tensor(librosa.power_to_db(mel_spec.squeeze().numpy()))
        mel_spec = self.amplitude_to_DB(mel_spec)
        mel_spec = mel_spec.squeeze()
        mel_spec = mel_spec.permute(1, 0) # (F, L) -> (L, F)
        this_mel_spec = self.denormalizer(this_mel_spec, mel_spec)
        return this_mel_spec
    
    def inverse(self, mel_spec): 
        mel_spec = mel_spec.permute(1, 0) # (L, F) -> (F, L)
        # mel_spec = torch.tensor(librosa.db_to_power(mel_spec.numpy()))
        mel_spec = torchaudio.functional.DB_to_amplitude(mel_spec, ref=1.0, power=1)
        mel_spec = mel_spec.unsqueeze(0)  # restore from (F, L) to (channel, F, L)
        i_mel = self.inverse_mel(mel_spec)
        inv = self.grifflim(i_mel)
        return inv
    

class MelSpecTransformNoDB(nn.Module): 
    def __init__(self, sample_rate, n_fft=400, n_mels=64, normalizer=None, denormalizer=None): 
        super().__init__()
        self.sample_rate = sample_rate
        n_stft = int((n_fft//2) + 1)
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=n_mels, n_fft=n_fft)
        self.inverse_mel = torchaudio.transforms.InverseMelScale(sample_rate=sample_rate, n_mels=n_mels, n_stft=n_stft)
        self.grifflim = torchaudio.transforms.GriffinLim(n_fft=n_fft)

        self.normalizer = normalizer
        self.denormalizer = denormalizer

    def forward(self, waveform): 
        # transform to mel_spectrogram
        mel_spec = self.transform(waveform)  # (channel, n_mels, time)
        mel_spec = mel_spec.squeeze()
        mel_spec = mel_spec.permute(1, 0) # (F, L) -> (L, F)
        mel_spec = self.normalizer(mel_spec)
        return mel_spec
    
    def de_norm(self, this_mel_spec, waveform): 
        # transform to mel_spectrogram
        mel_spec = self.transform(waveform)  # (channel, n_mels, time)
        mel_spec = mel_spec.squeeze()
        mel_spec = mel_spec.permute(1, 0) # (F, L) -> (L, F)
        this_mel_spec = self.denormalizer(this_mel_spec, mel_spec)
        return this_mel_spec
    
    def inverse(self, mel_spec): 
        mel_spec = mel_spec.permute(1, 0) # (L, F) -> (F, L)
        mel_spec = mel_spec.unsqueeze(0)  # restore from (F, L) to (channel, F, L)
        i_mel = self.inverse_mel(mel_spec)
        inv = self.grifflim(i_mel)
        return inv


class MelSpecTransform(nn.Module): 
    def __init__(self, sample_rate, n_fft=400, n_mels=64): 
        super().__init__()
        self.sample_rate = sample_rate
        n_stft = int((n_fft//2) + 1)
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=n_mels, n_fft=n_fft)
        self.inverse_mel = torchaudio.transforms.InverseMelScale(sample_rate=sample_rate, n_mels=n_mels, n_stft=n_stft)
        self.grifflim = torchaudio.transforms.GriffinLim(n_fft=n_fft)

    def forward(self, waveform): 
        # transform to mel_spectrogram
        mel_spec = self.transform(waveform)  # (channel, n_mels, time)
        mel_spec = mel_spec.squeeze()
        mel_spec = mel_spec.permute(1, 0) # (F, L) -> (L, F)

        """
        There should be normalization method here, 
        but for the moment we just leave it here, 
        later, consider PCEN
        """
        # # Apply normalization (CMVN)
        eps = 1e-9
        mean = mel_spec.mean()
        std = mel_spec.std(unbiased=False)
        # # print(feature.shape)
        # # print(mean, std)
        mel_spec = (mel_spec - mean) / (std + eps)

        # mel_spec = self.transform(waveform)
        # # mel_spec = self.to_db(mel_spec)
        # mel_spec = mel_spec.squeeze()
        # mel_spec = mel_spec.permute(1, 0) # (F, L) -> (L, F)
        return mel_spec
    
    def de_norm(self, this_mel_spec, waveform): 
        # transform to mel_spectrogram
        mel_spec = self.transform(waveform)  # (channel, n_mels, time)
        mel_spec = mel_spec.squeeze()
        mel_spec = mel_spec.permute(1, 0) # (F, L) -> (L, F)

        eps = 1e-9
        mean = mel_spec.mean()
        std = mel_spec.std(unbiased=False)

        this_mel_spec = this_mel_spec * std + mean
        return this_mel_spec
    
    def inverse(self, mel_spec): 
        mel_spec = mel_spec.permute(1, 0) # (L, F) -> (F, L)
        mel_spec = mel_spec.unsqueeze(0)  # restore from (F, L) to (channel, F, L)
        i_mel = self.inverse_mel(mel_spec)
        inv = self.grifflim(i_mel)
        return inv

class Normalizer(nn.Module):
    def __init__(self, fun): 
        super().__init__()
        self.fun = fun
    
    def forward(self, mel_spec):
        return self.fun(mel_spec)
    
    @staticmethod
    def norm_mvn(mel_spec):
        eps = 1e-9
        mean = mel_spec.mean()
        std = mel_spec.std(unbiased=False)
        norm_spec = (mel_spec - mean) / (std + eps)
        return norm_spec
    
    @staticmethod
    def norm_strip_mvn(mel_spec):
        eps = 1e-9
        mean = mel_spec.mean(1, keepdim=True)
        std = mel_spec.std(1, keepdim=True, unbiased=False)
        norm_spec = (mel_spec - mean) / (std + eps)
        return norm_spec
    
    @staticmethod
    def norm_time_mvn(mel_spec):
        # this is bad
        eps = 1e-9
        mean = mel_spec.mean(0, keepdim=True)
        std = mel_spec.std(0, keepdim=True, unbiased=False)
        norm_spec = (mel_spec - mean) / (std + eps)
        return norm_spec
    
    @staticmethod
    def norm_minmax(mel_spec):
        min_val = mel_spec.min()
        max_val = mel_spec.max()
        norm_spec = (mel_spec - min_val) / (max_val - min_val)
        return norm_spec
    
    @staticmethod
    def norm_strip_minmax(mel_spec):
        min_val = mel_spec.min(1, keepdim=True)[0]
        max_val = mel_spec.max(1, keepdim=True)[0]
        norm_spec = (mel_spec - min_val) / (max_val - min_val)
        return norm_spec
    
    @staticmethod
    def no_norm(mel_spec):
        return mel_spec


class DeNormalizer(nn.Module):
    def __init__(self, fun): 
        super().__init__()
        self.fun = fun
    
    def forward(self, mel_spec):
        return self.fun(mel_spec)
    
    @staticmethod
    def norm_mvn(mel_spec, non_normed_mel_spec):
        eps = 1e-9
        mean = non_normed_mel_spec.mean()
        std = non_normed_mel_spec.std(unbiased=False)

        mel_spec = mel_spec * std + mean
        return mel_spec
    
    @staticmethod
    def norm_strip_mvn(mel_spec, non_normed_mel_spec):
        eps = 1e-9
        mean = non_normed_mel_spec.mean(1, keepdim=True)
        std = non_normed_mel_spec.std(1, keepdim=True, unbiased=False)

        mel_spec = mel_spec * std + mean
        return mel_spec
    
    @staticmethod
    def no_norm(mel_spec, non_normed_mel_spec):
        return mel_spec
    
class MelSpecTransformOld(nn.Module): 
    def __init__(self, sample_rate, n_fft=400, n_mels=64): 
        super().__init__()
        self.sample_rate = sample_rate
        n_stft = int((n_fft//2) + 1)
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=n_mels, n_fft=n_fft)
        self.inverse_mel = torchaudio.transforms.InverseMelScale(sample_rate=sample_rate, n_mels=n_mels, n_stft=n_stft)
        self.grifflim = torchaudio.transforms.GriffinLim(n_fft=n_fft)
        
    
    def forward(self, waveform): 
        # transform to mel_spectrogram
        mel_spec = self.transform(waveform)  # (channel, n_mels, time)
        mel_spec = mel_spec.squeeze()
        mel_spec = mel_spec.permute(1, 0) # (F, L) -> (L, F)

        """
        There should be normalization method here, 
        but for the moment we just leave it here, 
        later, consider PCEN
        """
        # # Apply normalization (CMVN)
        eps = 1e-9
        mean = mel_spec.mean(0, keepdim=True)
        std = mel_spec.std(0, keepdim=True, unbiased=False)
        # # print(feature.shape)
        # # print(mean, std)
        mel_spec = (mel_spec - mean) / (std + eps)

        # mel_spec = self.transform(waveform)
        # # mel_spec = self.to_db(mel_spec)
        # mel_spec = mel_spec.squeeze()
        # mel_spec = mel_spec.permute(1, 0) # (F, L) -> (L, F)
        return mel_spec
    
    def de_norm(self, this_mel_spec, waveform): 
        # transform to mel_spectrogram
        mel_spec = self.transform(waveform)  # (channel, n_mels, time)
        mel_spec = mel_spec.squeeze()
        mel_spec = mel_spec.permute(1, 0) # (F, L) -> (L, F)

        eps = 1e-9
        mean = mel_spec.mean(0, keepdim=True)
        std = mel_spec.std(0, keepdim=True, unbiased=False)

        this_mel_spec = this_mel_spec * std + mean
        return this_mel_spec
    
    def inverse(self, mel_spec): 
        mel_spec = mel_spec.permute(1, 0) # (L, F) -> (F, L)
        mel_spec = mel_spec.unsqueeze(0)  # restore from (F, L) to (channel, F, L)
        i_mel = self.inverse_mel(mel_spec)
        inv = self.grifflim(i_mel)
        return inv



class MFCCTransform(nn.Module): 
    def __init__(self, sample_rate, n_fft): 
        super().__init__()
        # self.transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=n_fft, n_mels=64)
        # self.to_db = torchaudio.transforms.AmplitudeToDB()
        # self.transform = torchaudio.transforms.MFCC(n_mfcc=13)
    
    def forward(self, waveform, sr=16000): 
        # extract mfcc
        feature = torchaudio.compliance.kaldi.mfcc(waveform, sample_frequency=sr)

        # add deltas
        d1 = torchaudio.functional.compute_deltas(feature)
        d2 = torchaudio.functional.compute_deltas(d1)
        feature = torch.cat([feature, d1, d2], dim=-1)

        # Apply normalization (CMVN)
        eps = 1e-9
        mean = feature.mean(0, keepdim=True)
        std = feature.std(0, keepdim=True, unbiased=False)
        feature = (feature - mean) / (std + eps)
        return feature