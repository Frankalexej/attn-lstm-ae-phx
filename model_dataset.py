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
from misc_my_utils import time_to_frame
import torch.nn.functional as F
import pickle
from misc_tools import AudioCut
from misc_tools import PathUtils as PU
import numpy as np

class DS_Tools:
    @ staticmethod
    def save_indices(filename, my_list):
        try:
            with open(filename, 'wb') as file:
                pickle.dump(my_list, file)
            return True
        except Exception as e:
            print(f"An error occurred while saving the list: {e}")
            return False

    @ staticmethod    
    def read_indices(filename):
        try:
            with open(filename, 'rb') as file:
                my_list = pickle.load(file)
            return my_list
        except Exception as e:
            print(f"An error occurred while reading the list: {e}")
            return None 
    
    @staticmethod
    def create_ground_truth(length, bp_pair):
        # Initialize the ground truth tensor with zeros or a placeholder value
        ground_truth = torch.zeros(length, dtype=torch.int)

        # Start index for the first phoneme
        start_idx = 0

        # Process all but the last phoneme using the boundaries
        for (boundary, phoneme) in bp_pair[:-1]:
            ground_truth[start_idx:boundary] = phoneme
            start_idx = boundary

        # Handle the last phoneme, ensuring it extends to the end of the mel spectrogram if necessary
        ground_truth[start_idx:] = bp_pair[-1][0]

        return ground_truth

class TokenMap: 
    def __init__(self, token_list, starter=0):  
        self.token2idx = {element: index + starter for index, element in enumerate(token_list)}
        self.idx2token = {index + starter: element for index, element in enumerate(token_list)}
    
    def encode(self, token): 
        return self.token2idx[token]
    
    def decode(self, idx): 
        return self.idx2token[idx]
    
    def token_num(self): 
        return len(self.token2idx)


class WordDataset(Dataset):
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None):
        if isinstance(guide_, str):
            guide_file = pd.read_csv(guide_)
        elif isinstance(guide_, pd.DataFrame):
            guide_file = guide_
        else:
            raise Exception("Guide neither to read or to be used directly")

        # guide_file = guide_file[guide_file["segment_nostress"].isin(select)]
        # this is not needed for worddataset, we only need to kick out the non-word segments
        guide_file = guide_file[~guide_file["segment_nostress"].isin(["sil", "sp", "spn"])]
        guide_file = guide_file[guide_file['word_nSample'] > 400]
        guide_file = guide_file[guide_file['word_nSample'] <= 15000]
        
        path_col = guide_file["word_path"].unique()
        # have to use unique, because we are working on matched_phone_guide, 
        # which repetitively marks the word path if the segment belongs to the word
        
        # seg_col = guide_file["segment_nostress"]
        
        self.guide_file = guide_file
        self.dataset = path_col.tolist()
        self.src_dir = src_dir
        self.transform = transform

        self.mapper = mapper
        # should not be used unless for derived classes that use ground truth. 
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_name = os.path.join(
            self.src_dir, 
            self.dataset[idx]
        )
        
        data, sample_rate = torchaudio.load(file_name, normalize=True)
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
    
class WordDictionary:
    def __init__(self, file_path):
        with open(file_path, 'rb') as file:
            unique_words_list = pickle.load(file)
        self.word2idx = {word: idx for idx, word in enumerate(unique_words_list)}
        self.idx2word = {idx: word for idx, word in enumerate(unique_words_list)}
    
    def encode(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return 0
    
    def decode(self, idx): 
        return self.idx2word[idx]
    
    def token_num(self): 
        return len(self.word2idx)
    
class WordDatasetWord(WordDataset): 
    # This is for WIDAE. We additionally return word. 
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None):
        super().__init__(src_dir, guide_, select, mapper, transform)
        self.word_set = self.guide_file.drop_duplicates(subset='wuid', keep='first')["word"].tolist()
        self.mapper = mapper
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        word = self.word_set[idx]
        
        return data, self.mapper.encode(word)

    @staticmethod
    def collate_fn(data):
        xx, word = zip(*data)
        # only working for one data at the moment
        batch_first = True
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return xx_pad, x_lens, word
    
class WordDatasetWordPath(WordDatasetWord): 
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None):
        super().__init__(src_dir, guide_, select, mapper, transform)
        self.name_set = self.guide_file["wuid"].unique().tolist()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, word = super().__getitem__(idx)
        name = self.name_set[idx]
        
        return data, word, name

    @staticmethod
    def collate_fn(data):
        xx, word, name = zip(*data)
        # only working for one data at the moment
        batch_first = True
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return xx_pad, x_lens, word, name

class WordDatasetPath(WordDataset): 
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None):
        super().__init__(src_dir, guide_, select, mapper, transform)
        self.name_set = self.guide_file["wuid"].unique().tolist()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
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
    
class WordDatasetRandomMasking(WordDataset): 
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None, mask_rate=0.15):
        super().__init__(src_dir, guide_, select, mapper, transform)
        self.mask_rate = mask_rate
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        # this is only applicable with transform. 
        # this version does not apply continuous selection, just random masking
        l, d = data.shape   # length, dimension
        mask = torch.rand((l, )) > self.mask_rate
        masked_data = data * mask.unsqueeze(-1)
        return masked_data, data

    @staticmethod
    def collate_fn(data):
        masked_xx, xx = zip(*data)
        batch_first = True
        masked_x_lens = [len(x) for x in masked_xx]
        masked_xx_pad = pad_sequence(masked_xx, batch_first=batch_first, padding_value=0)
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return masked_xx_pad, masked_x_lens, xx_pad, x_lens

class WordDatasetFramephone(WordDataset):
    """
    WordDataset with paired boundary information. 
    Notice that in the matched phone guide there are naturally phones with starting and ending times marked. 
    We also know to which word each phone belongs to. 
    So we want to start from here and calculated the frames of each phone's ending times. 
    """
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None, ground_truth_path=""):
        super().__init__(src_dir, guide_, select, mapper, transform)

        self.mapper = mapper

        if ground_truth_path != "" and PU.path_exist(ground_truth_path): 
            with open(ground_truth_path, "rb") as file:
                # Load the object from the file
                self.ground_truth_set = pickle.load(file)
        else: 
            # e1/e2/.../en (belong to same word) -> [t1, t2, ..., tn] -> [f1, f2, ..., fn]
            bp_pair_set = self.guide_file.groupby('wuid').apply(lambda x: [(self.mapper.encode(row["segment_nostress"]), time_to_frame(row['endTime'] - row['word_startTime'])) for index, row in x.iterrows()])
            unique_wuid_df = self.guide_file.drop_duplicates(subset='wuid')
            word_framelength_set = ((unique_wuid_df['word_nSample'] // 200).astype(int) + 1).tolist()

            self.ground_truth_set = [DS_Tools.create_ground_truth(word_framelength, 
                                                                bp_pair) for 
                                    word_framelength, bp_pair in 
                                    zip(word_framelength_set, bp_pair_set)]
        
            with open(ground_truth_path, 'wb') as file:
                pickle.dump(self.ground_truth_set, file)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        # extra info for completing a csv
        ground_truth = self.ground_truth_set[idx]
        
        return data, ground_truth
    
    @staticmethod
    def collate_fn(data):
        # xx = data, aa bb cc = info_rec, info_idx, info_token
        xx, yy = zip(*data)
        # only working for one data at the moment
        batch_first = True
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        y_lens = [len(x) for x in yy]
        yy_pad = pad_sequence(yy, batch_first=batch_first, padding_value=0)
        return xx_pad, x_lens, yy_pad, y_lens
    
class WordDatasetPhoneseq(WordDataset):
    """
    This will pair each rec with the corresponding sequence of phones it contains. 
    Notice that is given as a list of numbers, which is translated from the phoneme letter-spellings. 
    """
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None, ground_truth_path=""):
        super().__init__(src_dir, guide_, select, mapper, transform)

        self.mapper = mapper

        if ground_truth_path != "" and PU.path_exist(ground_truth_path): 
            with open(ground_truth_path, "rb") as file:
                # Load the object from the file
                self.ground_truth_set = pickle.load(file)
                # self.ground_truth_set = [torch.tensor(one_gt) for one_gt in self.ground_truth_set]
                # with open(ground_truth_path, 'wb') as file:
                #     pickle.dump(self.ground_truth_set, file)
        else: 
            # e1/e2/.../en (belong to same word) -> [t1, t2, ..., tn] -> [f1, f2, ..., fn]
            self.ground_truth_set = self.guide_file.groupby('wuid').apply(lambda x: torch.tensor([self.mapper.encode(row["segment_nostress"]) for index, row in x.iterrows()])).tolist()
        
            with open(ground_truth_path, 'wb') as file:
                pickle.dump(self.ground_truth_set, file)

        assert len(self.dataset) == len(self.ground_truth_set)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        # extra info for completing a csv
        ground_truth = self.ground_truth_set[idx]
        
        return data, ground_truth
    
    @staticmethod
    def collate_fn(data):
        # xx = data, aa bb cc = info_rec, info_idx, info_token
        xx, yy = zip(*data)
        # only working for one data at the moment
        batch_first = True
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        y_lens = [len(x) for x in yy]
        yy_pad = pad_sequence(yy, batch_first=batch_first, padding_value=0)
        # yy_cat = torch.cat(yy)
        # NOTE: this is to be consistent with TargetVowel Dataset
        return (xx_pad, yy_pad), (x_lens, y_lens), None, None
    
    @staticmethod
    def collate_fn_yNoPad(data):
        # xx = data, aa bb cc = info_rec, info_idx, info_token
        xx, yy = zip(*data)
        # only working for one data at the moment
        batch_first = True
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        y_lens = [len(x) for x in yy]
        # yy_pad = pad_sequence(yy, batch_first=batch_first, padding_value=0)
        # yy_cat = torch.cat(yy)
        return (xx_pad, yy), (x_lens, y_lens), None, None

class WordDatasetBoundary(WordDataset):
    """
    WordDataset with paired framephone information. 
    Framephone is the nickname for the ground truth inferred from the matched phone guide, which marks which phones belong to the current word. 
    Now we take this, also given the boundary generate a vector of shape same as the melspectrogram of words. Each frame a phone. 
    """
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None):
        super().__init__(src_dir, guide_, select, mapper, transform)
        self.name_set = self.guide_file["wuid"].tolist()

        # e1/e2/.../en (belong to same word) -> [t1, t2, ..., tn] -> [f1, f2, ..., fn]
        self.bnd_set = self.guide_file.groupby('wuid').apply(lambda x: [time_to_frame(row['endTime'] - row['word_startTime']) for index, row in x.iterrows()]).tolist()
        
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        # extra info for completing a csv
        bnd = self.bnd_set[idx]
        name = self.name_set[idx]
        
        return data, bnd, name
    
    @staticmethod
    def collate_fn(data):
        # xx = data, aa bb cc = info_rec, info_idx, info_token
        xx, bnd, name = zip(*data)
        # only working for one data at the moment
        batch_first = True
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return xx_pad, x_lens, bnd, name
    

########################### Phenomenon-Target Training Dataset ###########################
class TargetDataset(Dataset):
    # Target means the phenomenon-target, that is, e.g. /th/ or /st/. 
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None):
        # guide_file = pd.read_csv(guide_)
        if isinstance(guide_, str):
            guide_file = pd.read_csv(guide_)
        elif isinstance(guide_, pd.DataFrame):
            guide_file = guide_
        else:
            raise Exception("Guide neither to read or to be used directly")

        # guide_file = guide_file[guide_file["segment_nostress"].isin(select)]
        # this is not needed for worddataset, we only need to kick out the non-word segments
        # guide_file = guide_file[~guide_file["segment_nostress"].isin(["sil", "sp", "spn"])]
        # guide_file = guide_file[guide_file['word_nSample'] > 400]
        # guide_file = guide_file[guide_file['word_nSample'] <= 15000]
        
        sib_path_col = guide_file["sibilant_path"]
        stop_path_col = guide_file["stop_path"]
        phi_type_col = guide_file["phi_type"]
        stop_name_col = guide_file["stop"]
        
        self.guide_file = guide_file
        self.dataset = stop_path_col.tolist()
        self.sib_path = sib_path_col.tolist()
        self.phi_type = phi_type_col.tolist()
        self.stop_name = stop_name_col.tolist()
        self.src_dir = src_dir
        self.transform = transform

        self.mapper = mapper
        # should not be used unless for derived classes that use ground truth. 
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.phi_type[idx] == "ST": 
            # read two and concat
            S_name = os.path.join(
                self.src_dir, 
                self.sib_path[idx]
            )
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )

            S_data, sample_rate_S = torchaudio.load(S_name, normalize=True)
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            assert sample_rate_S == sample_rate_T

            data = torch.cat([S_data, T_data], dim=1)
        else: 
            # "T"
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
        
            data, sample_rate = torchaudio.load(T_name, normalize=True)

        if self.transform:
            data = self.transform(data)
        
        return data, self.phi_type[idx], self.stop_name[idx]

    @staticmethod
    def collate_fn(data):
        # only working for one data at the moment
        batch_first = True
        xx, pt, sn = zip(*data)
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return xx_pad, x_lens, pt, sn            


class TargetDatasetBoundary(Dataset):
    # Target means the phenomenon-target, that is, e.g. /th/ or /st/. 
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None):
        # guide_file = pd.read_csv(guide_)
        if isinstance(guide_, str):
            guide_file = pd.read_csv(guide_)
        elif isinstance(guide_, pd.DataFrame):
            guide_file = guide_
        else:
            raise Exception("Guide neither to read or to be used directly")

        # guide_file = guide_file[guide_file["segment_nostress"].isin(select)]
        # this is not needed for worddataset, we only need to kick out the non-word segments
        # guide_file = guide_file[~guide_file["segment_nostress"].isin(["sil", "sp", "spn"])]
        # guide_file = guide_file[guide_file['word_nSample'] > 400]
        # guide_file = guide_file[guide_file['word_nSample'] <= 15000]
        guide_file["sep_frame"] = guide_file.apply(lambda x: time_to_frame(x['stop_startTime'] - x['sibilant_startTime']), axis=1)
        
        sib_path_col = guide_file["sibilant_path"]
        stop_path_col = guide_file["stop_path"]
        phi_type_col = guide_file["phi_type"]
        stop_name_col = guide_file["stop"]
        sep_frame_col = guide_file["sep_frame"]
        
        self.guide_file = guide_file
        self.dataset = stop_path_col.tolist()
        self.sib_path = sib_path_col.tolist()
        self.phi_type = phi_type_col.tolist()
        self.stop_name = stop_name_col.tolist()
        self.sep_frame = sep_frame_col.tolist()
        self.src_dir = src_dir
        self.transform = transform

        self.mapper = mapper
        # should not be used unless for derived classes that use ground truth. 
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.phi_type[idx] == "ST": 
            # read two and concat
            S_name = os.path.join(
                self.src_dir, 
                self.sib_path[idx]
            )
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )

            S_data, sample_rate_S = torchaudio.load(S_name, normalize=True)
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            assert sample_rate_S == sample_rate_T

            data = torch.cat([S_data, T_data], dim=1)
        else: 
            # "T"
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
        
            data, sample_rate = torchaudio.load(T_name, normalize=True)

        if self.transform:
            data = self.transform(data)
        
        return data, self.phi_type[idx], self.stop_name[idx], self.sep_frame[idx]

    @staticmethod
    def collate_fn(data):
        # only working for one data at the moment
        batch_first = True
        xx, pt, sn, sf = zip(*data)
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return xx_pad, x_lens, pt, sn, sf


########################### Phenomenon-Target Training Dataset (Vowel-included) ###########################
class TargetVowelDataset(Dataset):
    # Target means the phenomenon-target, that is, e.g. /th/ or /st/. 
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None):
        # guide_file = pd.read_csv(guide_)
        if isinstance(guide_, str):
            guide_file = pd.read_csv(guide_)
        elif isinstance(guide_, pd.DataFrame):
            guide_file = guide_
        else:
            raise Exception("Guide neither to read or to be used directly")
        
        pre_path_col = guide_file["pre_path"]
        stop_path_col = guide_file["stop_path"]
        vowel_path_col = guide_file["vowel_path"]

        phi_type_col = guide_file["phi_type"]
        stop_name_col = guide_file["stop"]
        
        self.guide_file = guide_file
        self.dataset = stop_path_col.tolist()
        self.pre_path = pre_path_col.tolist()
        self.vowel_path = vowel_path_col.tolist()
        self.phi_type = phi_type_col.tolist()
        self.stop_name = stop_name_col.tolist()
        self.src_dir = src_dir
        self.transform = transform

        self.mapper = mapper
        # should not be used unless for derived classes that use ground truth. 
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.phi_type[idx] == "ST": 
            # read two and concat
            S_name = os.path.join(
                self.src_dir, 
                self.pre_path[idx]
            )
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
            V_name = os.path.join(
                self.src_dir, 
                self.vowel_path[idx]
            )

            S_data, sample_rate_S = torchaudio.load(S_name, normalize=True)
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            V_data, sample_rate_V = torchaudio.load(V_name, normalize=True)
            assert sample_rate_S == sample_rate_T == sample_rate_V

            data = torch.cat([S_data, T_data, V_data], dim=1)
        else: 
            # "T"
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
            V_name = os.path.join(
                self.src_dir, 
                self.vowel_path[idx]
            )
        
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            V_data, sample_rate_V = torchaudio.load(V_name, normalize=True)
            assert sample_rate_T == sample_rate_V

            data = torch.cat([T_data, V_data], dim=1)

        if self.transform:
            data = self.transform(data)
        
        return data, self.phi_type[idx], self.stop_name[idx]

    @staticmethod
    def collate_fn(data):
        # only working for one data at the moment
        batch_first = True
        xx, pt, sn = zip(*data)
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return xx_pad, x_lens, pt, sn
    
class TargetVowelDatasetWord(Dataset): 
    # Target means the phenomenon-target, that is, e.g. /th/ or /st/. 
    # This dataset additionally returns the word. 
    # Let's use the select field to pass in the word list.
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None):
        # guide_file = pd.read_csv(guide_)
        if isinstance(guide_, str):
            guide_file = pd.read_csv(guide_)
        elif isinstance(guide_, pd.DataFrame):
            guide_file = guide_
        else:
            raise Exception("Guide neither to read or to be used directly")
        
        if isinstance(select, str):
            word_file = pd.read_csv(select)
        elif isinstance(select, pd.DataFrame):
            word_file = select
        else:
            raise Exception("Guide neither to read or to be used directly")
        
        pre_path_col = guide_file["pre_path"]
        stop_path_col = guide_file["stop_path"]
        vowel_path_col = guide_file["vowel_path"]

        phi_type_col = guide_file["phi_type"]
        stop_name_col = guide_file["stop"]

        # get the corresponding word
        unique_word = word_file.drop_duplicates(subset='wuid', keep='first')
        wuid_col = guide_file["wuid"]
        wuid_to_word = pd.Series(unique_word['word'].values, index=unique_word["wuid"])
        self.word_set = wuid_col.map(wuid_to_word).tolist()
        
        self.guide_file = guide_file
        self.dataset = stop_path_col.tolist()
        self.pre_path = pre_path_col.tolist()
        self.vowel_path = vowel_path_col.tolist()
        self.phi_type = phi_type_col.tolist()
        self.stop_name = stop_name_col.tolist()
        self.src_dir = src_dir
        self.transform = transform

        self.mapper = mapper
        # should not be used unless for derived classes that use ground truth. 
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.phi_type[idx] == "ST": 
            # read two and concat
            S_name = os.path.join(
                self.src_dir, 
                self.pre_path[idx]
            )
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
            V_name = os.path.join(
                self.src_dir, 
                self.vowel_path[idx]
            )

            S_data, sample_rate_S = torchaudio.load(S_name, normalize=True)
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            V_data, sample_rate_V = torchaudio.load(V_name, normalize=True)
            assert sample_rate_S == sample_rate_T == sample_rate_V

            data = torch.cat([S_data, T_data, V_data], dim=1)
        else: 
            # "T"
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
            V_name = os.path.join(
                self.src_dir, 
                self.vowel_path[idx]
            )
        
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            V_data, sample_rate_V = torchaudio.load(V_name, normalize=True)
            assert sample_rate_T == sample_rate_V

            data = torch.cat([T_data, V_data], dim=1)

        if self.transform:
            data = self.transform(data)
        
        return data, self.phi_type[idx], self.stop_name[idx], self.mapper.encode(self.word_set[idx])

    @staticmethod
    def collate_fn(data):
        # only working for one data at the moment
        batch_first = True
        xx, pt, sn, word = zip(*data)
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return xx_pad, x_lens, pt, sn, word


class TargetVowelDatasetBoundary(Dataset): 
    # this mixes T and ST. 
    # Target means the phenomenon-target, that is, e.g. /th/ or /st/. 
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None):
        # guide_file = pd.read_csv(guide_)
        if isinstance(guide_, str):
            guide_file = pd.read_csv(guide_)
        elif isinstance(guide_, pd.DataFrame):
            guide_file = guide_
        else:
            raise Exception("Guide neither to read or to be used directly")
        
        guide_file["first_sep_frame"] = guide_file.apply(lambda x: time_to_frame(x['stop_startTime'] - x['pre_startTime']), axis=1)
        guide_file["second_sep_frame"] = guide_file.apply(lambda x: time_to_frame(x['vowel_startTime'] - x['pre_startTime']), axis=1)
        
        pre_path_col = guide_file["pre_path"]
        stop_path_col = guide_file["stop_path"]
        vowel_path_col = guide_file["vowel_path"]
        phi_type_col = guide_file["phi_type"]
        stop_name_col = guide_file["stop"]
        vowel_name_col = guide_file["vowel"]
        first_sep_frame_col = guide_file["first_sep_frame"]
        second_sep_frame_col = guide_file["second_sep_frame"]
        
        self.guide_file = guide_file
        self.dataset = stop_path_col.tolist()
        self.pre_path = pre_path_col.tolist()
        self.vowel_path = vowel_path_col.tolist()
        self.phi_type = phi_type_col.tolist()
        self.stop_name = stop_name_col.tolist()
        self.vowel_name = vowel_name_col.tolist()
        self.first_sep_frame = first_sep_frame_col.tolist()
        self.second_sep_frame = second_sep_frame_col.tolist()
        self.src_dir = src_dir
        self.transform = transform

        self.mapper = mapper
        # should not be used unless for derived classes that use ground truth. 
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.phi_type[idx] == "ST": 
            # read two and concat
            S_name = os.path.join(
                self.src_dir, 
                self.pre_path[idx]
            )
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
            V_name = os.path.join(
                self.src_dir, 
                self.vowel_path[idx]
            )

            S_data, sample_rate_S = torchaudio.load(S_name, normalize=True)
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            V_data, sample_rate_V = torchaudio.load(V_name, normalize=True)
            assert sample_rate_S == sample_rate_T == sample_rate_V

            data = torch.cat([S_data, T_data, V_data], dim=1)
        else: 
            # "T"
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
            V_name = os.path.join(
                self.src_dir, 
                self.vowel_path[idx]
            )
        
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            V_data, sample_rate_V = torchaudio.load(V_name, normalize=True)
            assert sample_rate_T == sample_rate_V

            data = torch.cat([T_data, V_data], dim=1)

        if self.transform:
            data = self.transform(data)
        
        return data, self.phi_type[idx], self.stop_name[idx], self.vowel_name[idx], self.first_sep_frame[idx], self.second_sep_frame[idx]

    @staticmethod
    def collate_fn(data):
        # only working for one data at the moment
        batch_first = True
        xx, pt, sn, vn, sf1, sf2 = zip(*data)
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return xx_pad, x_lens, pt, sn, vn, sf1, sf2
    
class TargetVowelDatasetBoundaryWord(Dataset): 
    # this mixes T and ST. 
    # Target means the phenomenon-target, that is, e.g. /th/ or /st/. 
    # also get the word id
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None):
        # guide_file = pd.read_csv(guide_)
        if isinstance(guide_, str):
            guide_file = pd.read_csv(guide_)
        elif isinstance(guide_, pd.DataFrame):
            guide_file = guide_
        else:
            raise Exception("Guide neither to read or to be used directly")
        if isinstance(select, str):
            word_file = pd.read_csv(select)
        elif isinstance(select, pd.DataFrame):
            word_file = select
        else:
            raise Exception("Guide neither to read or to be used directly")
        
        guide_file["first_sep_frame"] = guide_file.apply(lambda x: time_to_frame(x['stop_startTime'] - x['pre_startTime']), axis=1)
        guide_file["second_sep_frame"] = guide_file.apply(lambda x: time_to_frame(x['vowel_startTime'] - x['pre_startTime']), axis=1)
        
        pre_path_col = guide_file["pre_path"]
        stop_path_col = guide_file["stop_path"]
        vowel_path_col = guide_file["vowel_path"]
        phi_type_col = guide_file["phi_type"]
        stop_name_col = guide_file["stop"]
        vowel_name_col = guide_file["vowel"]
        first_sep_frame_col = guide_file["first_sep_frame"]
        second_sep_frame_col = guide_file["second_sep_frame"]

        # get the corresponding word
        unique_word = word_file.drop_duplicates(subset='wuid', keep='first')
        wuid_col = guide_file["wuid"]
        wuid_to_word = pd.Series(unique_word['word'].values, index=unique_word["wuid"])
        self.word_set = wuid_col.map(wuid_to_word).tolist()
        
        self.guide_file = guide_file
        self.dataset = stop_path_col.tolist()
        self.pre_path = pre_path_col.tolist()
        self.vowel_path = vowel_path_col.tolist()
        self.phi_type = phi_type_col.tolist()
        self.stop_name = stop_name_col.tolist()
        self.vowel_name = vowel_name_col.tolist()
        self.first_sep_frame = first_sep_frame_col.tolist()
        self.second_sep_frame = second_sep_frame_col.tolist()
        self.src_dir = src_dir
        self.transform = transform

        self.mapper = mapper
        # should not be used unless for derived classes that use ground truth. 
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.phi_type[idx] == "ST": 
            # read two and concat
            S_name = os.path.join(
                self.src_dir, 
                self.pre_path[idx]
            )
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
            V_name = os.path.join(
                self.src_dir, 
                self.vowel_path[idx]
            )

            S_data, sample_rate_S = torchaudio.load(S_name, normalize=True)
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            V_data, sample_rate_V = torchaudio.load(V_name, normalize=True)
            assert sample_rate_S == sample_rate_T == sample_rate_V

            data = torch.cat([S_data, T_data, V_data], dim=1)
        else: 
            # "T"
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
            V_name = os.path.join(
                self.src_dir, 
                self.vowel_path[idx]
            )
        
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            V_data, sample_rate_V = torchaudio.load(V_name, normalize=True)
            assert sample_rate_T == sample_rate_V

            data = torch.cat([T_data, V_data], dim=1)

        if self.transform:
            data = self.transform(data)
        
        return data, self.phi_type[idx], self.stop_name[idx], self.vowel_name[idx], self.first_sep_frame[idx], self.second_sep_frame[idx], self.mapper.encode(self.word_set[idx])

    @staticmethod
    def collate_fn(data):
        # only working for one data at the moment
        batch_first = True
        xx, pt, sn, vn, sf1, sf2, word = zip(*data)
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return xx_pad, x_lens, pt, sn, vn, sf1, sf2, word





#######################################################################################################
################################### TargetVowelDataset for Phoneseq ###################################
# Defs
class SilenceSampler_for_TV: 
    def __init__(self): 
        # mean and std fixed calculated from ST dataset. 
        mean = 0.09068133728311471
        std = 0.03271727752124411

        self.sigma = np.sqrt(np.log(1 + (std/mean)**2))
        self.mu = np.log(mean) - 0.5 * self.sigma**2
    
    def sample(self, size): 
        samples = np.random.lognormal(mean=self.mu, sigma=self.sigma, size=size)
        return samples
    

class WhiteNoiseGen: 
    def __init__(self, sample_rate, amplitude_scale):
        self.sr = sample_rate
        self.amplitude_scale = amplitude_scale
    
    def generate(self, duration): 
        num_samples = int(duration * self.sr)
        noise = torch.randn(num_samples) * self.amplitude_scale
        noise = noise.unsqueeze(0)
        return noise, self.sr
    
    def generate_samples(self, durations: np.array, samples=1): 
        num_samples = (durations * self.sr).astype(int)
        noises = [torch.randn((1, num_sample)) * self.amplitude_scale for num_sample in num_samples]
        return noises


class TargetVowelDatasetPhoneseq(Dataset): 
    # Target means the phenomenon-target, that is, e.g. /th/ or /st/. 
    # This dataset additionally returns the phone seq. 
    # NOTE: for TV condition we add silence as # in the place of S
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None, plosive_suffix=""):
        # guide_file = pd.read_csv(guide_)
        if isinstance(guide_, str):
            guide_file = pd.read_csv(guide_)
        elif isinstance(guide_, pd.DataFrame):
            guide_file = guide_
        else:
            raise Exception("Guide neither to read or to be used directly")
        
        # ""即不區分ST和T，"H"則是區分
        self.plosive_suffix = plosive_suffix
        
        pre_path_col = guide_file["pre_path"]
        stop_path_col = guide_file["stop_path"]
        vowel_path_col = guide_file["vowel_path"]

        phi_type_col = guide_file["phi_type"]
        stop_name_col = guide_file["stop"]
        vowel_name_col = guide_file["vowel"]

        # generate random length silence for TV condition
        self.silence_duration = SilenceSampler_for_TV().sample(len(stop_name_col))
        
        self.guide_file = guide_file
        self.dataset = stop_path_col.tolist()
        self.pre_path = pre_path_col.tolist()
        self.vowel_path = vowel_path_col.tolist()
        self.phi_type = phi_type_col.tolist()
        self.stop_name = stop_name_col.tolist()
        self.vowel_name = vowel_name_col.tolist()
        self.src_dir = src_dir
        self.transform = transform

        self.mapper = mapper
        noise_gen = WhiteNoiseGen(sample_rate=16000, amplitude_scale=0.01)
        self.noise_set = noise_gen.generate_samples(np.array(self.silence_duration))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.phi_type[idx] == "ST": 
            # read two and concat
            S_name = os.path.join(
                self.src_dir, 
                self.pre_path[idx]
            )
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
            V_name = os.path.join(
                self.src_dir, 
                self.vowel_path[idx]
            )

            S_data, sample_rate_S = torchaudio.load(S_name, normalize=True)
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            V_data, sample_rate_V = torchaudio.load(V_name, normalize=True)
            assert sample_rate_S == sample_rate_T == sample_rate_V

            data = torch.cat([S_data, T_data, V_data], dim=1)
            phoneseq = torch.tensor([self.mapper.encode(segment) for segment in ['S', self.stop_name[idx], self.vowel_name[idx]]], 
                                    dtype=torch.long)
        else: 
            # "T"
            # NOTE: in this way we equate STV and #TV conditions and they are now directly comparable. 
            Sil_duration = self.silence_duration[idx]
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
            V_name = os.path.join(
                self.src_dir, 
                self.vowel_path[idx]
            )
            Sil_data = self.noise_set[idx]
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            V_data, sample_rate_V = torchaudio.load(V_name, normalize=True)
            assert sample_rate_T == sample_rate_V

            data = torch.cat([Sil_data, T_data, V_data], dim=1)
            phoneseq = torch.tensor([self.mapper.encode(segment) for segment in [self.stop_name[idx] + self.plosive_suffix, self.vowel_name[idx]]], 
                                    dtype=torch.long)   # 'SIL', 

        if self.transform:
            data = self.transform(data)
        
        return data, self.phi_type[idx], self.stop_name[idx], phoneseq

    @staticmethod
    def collate_fn(data):
        # only working for one data at the moment
        batch_first = True
        xx, phi_type, stop_name, phoneseq = zip(*data)
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        phoneseq_lens = [len(x) for x in phoneseq]
        phoneseq_pad = pad_sequence(phoneseq, batch_first=batch_first, padding_value=0)
        return (xx_pad, phoneseq_pad), (x_lens, phoneseq_lens), phi_type, stop_name

class TargetVowelDatasetPhoneseqBothSIL(Dataset): 
    # Target means the phenomenon-target, that is, e.g. /th/ or /st/. 
    # This dataset additionally returns the phone seq. 
    # NOTE: for TV condition we add silence as # in the place of S
    # NOTE: For this version, we add silence in both ST and T conditions. 
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None, plosive_suffix=""):
        # guide_file = pd.read_csv(guide_)
        if isinstance(guide_, str):
            guide_file = pd.read_csv(guide_)
        elif isinstance(guide_, pd.DataFrame):
            guide_file = guide_
        else:
            raise Exception("Guide neither to read or to be used directly")
        
        # ""即不區分ST和T，"H"則是區分
        self.plosive_suffix = plosive_suffix
        
        pre_path_col = guide_file["pre_path"]
        stop_path_col = guide_file["stop_path"]
        vowel_path_col = guide_file["vowel_path"]

        phi_type_col = guide_file["phi_type"]
        stop_name_col = guide_file["stop"]
        vowel_name_col = guide_file["vowel"]

        # generate random length silence for TV condition
        self.silence_duration = SilenceSampler_for_TV().sample(len(stop_name_col))
        
        self.guide_file = guide_file
        self.dataset = stop_path_col.tolist()
        self.pre_path = pre_path_col.tolist()
        self.vowel_path = vowel_path_col.tolist()
        self.phi_type = phi_type_col.tolist()
        self.stop_name = stop_name_col.tolist()
        self.vowel_name = vowel_name_col.tolist()
        self.src_dir = src_dir
        self.transform = transform

        self.mapper = mapper
        noise_gen = WhiteNoiseGen(sample_rate=16000, amplitude_scale=0.01)
        self.noise_set = noise_gen.generate_samples(np.array(self.silence_duration))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.phi_type[idx] == "ST": 
            # read two and concat
            S_name = os.path.join(
                self.src_dir, 
                self.pre_path[idx]
            )
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
            V_name = os.path.join(
                self.src_dir, 
                self.vowel_path[idx]
            )

            # here we also add silence
            Sil_data = self.noise_set[idx]
            S_data, sample_rate_S = torchaudio.load(S_name, normalize=True)
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            V_data, sample_rate_V = torchaudio.load(V_name, normalize=True)
            assert sample_rate_S == sample_rate_T == sample_rate_V

            data = torch.cat([Sil_data, S_data, T_data, V_data], dim=1)
            phoneseq = torch.tensor([self.mapper.encode(segment) for segment in ['S', self.stop_name[idx], self.vowel_name[idx]]], 
                                    dtype=torch.long)
        else: 
            # "T"
            # NOTE: in this way we equate STV and #TV conditions and they are now directly comparable. 
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
            V_name = os.path.join(
                self.src_dir, 
                self.vowel_path[idx]
            )
            Sil_data = self.noise_set[idx]
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            V_data, sample_rate_V = torchaudio.load(V_name, normalize=True)
            assert sample_rate_T == sample_rate_V

            data = torch.cat([Sil_data, T_data, V_data], dim=1)
            phoneseq = torch.tensor([self.mapper.encode(segment) for segment in [self.stop_name[idx] + self.plosive_suffix, self.vowel_name[idx]]], 
                                    dtype=torch.long)   # 'SIL', 

        if self.transform:
            data = self.transform(data)
        
        return data, self.phi_type[idx], self.stop_name[idx], phoneseq

    @staticmethod
    def collate_fn(data):
        # only working for one data at the moment
        batch_first = True
        xx, phi_type, stop_name, phoneseq = zip(*data)
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        phoneseq_lens = [len(x) for x in phoneseq]
        phoneseq_pad = pad_sequence(phoneseq, batch_first=batch_first, padding_value=0)
        return (xx_pad, phoneseq_pad), (x_lens, phoneseq_lens), phi_type, stop_name



class TargetVowelDatasetBoundaryPhoneseq(Dataset): 
    # this mixes T and ST. 
    # Target means the phenomenon-target, that is, e.g. /th/ or /st/. 
    # NOTE: for TV condition we add silence as # in the place of S
    # NOTE: this version does not generate random silence here, because for evaluation 
    # NOTE: silence can be generated outside. 
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None, plosive_suffix=""):
        # guide_file = pd.read_csv(guide_)
        if isinstance(guide_, str):
            guide_file = pd.read_csv(guide_)
        elif isinstance(guide_, pd.DataFrame):
            guide_file = guide_
        else:
            raise Exception("Guide neither to read or to be used directly")
        
        self.plosive_suffix = plosive_suffix
        
        guide_file["first_sep_frame"] = guide_file.apply(lambda x: time_to_frame(x['stop_startTime'] - x['pre_startTime']), axis=1)
        guide_file["second_sep_frame"] = guide_file.apply(lambda x: time_to_frame(x['vowel_startTime'] - x['pre_startTime']), axis=1)
        
        pre_path_col = guide_file["pre_path"]
        stop_path_col = guide_file["stop_path"]
        vowel_path_col = guide_file["vowel_path"]
        phi_type_col = guide_file["phi_type"]
        stop_name_col = guide_file["stop"]
        vowel_name_col = guide_file["vowel"]
        first_sep_frame_col = guide_file["first_sep_frame"]
        second_sep_frame_col = guide_file["second_sep_frame"]
        
        # generate random length silence for TV condition
        self.silence_duration = (guide_file['stop_startTime'] - guide_file['pre_startTime']).tolist()

        self.guide_file = guide_file
        self.dataset = stop_path_col.tolist()
        self.pre_path = pre_path_col.tolist()
        self.vowel_path = vowel_path_col.tolist()
        self.phi_type = phi_type_col.tolist()
        self.stop_name = stop_name_col.tolist()
        self.vowel_name = vowel_name_col.tolist()
        self.first_sep_frame = first_sep_frame_col.tolist()
        self.second_sep_frame = second_sep_frame_col.tolist()
        self.src_dir = src_dir
        self.transform = transform

        self.mapper = mapper
        noise_gen = WhiteNoiseGen(sample_rate=16000, amplitude_scale=0.01)
        self.noise_set = noise_gen.generate_samples(np.array(self.silence_duration))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.phi_type[idx] == "ST": 
            # read two and concat
            S_name = os.path.join(
                self.src_dir, 
                self.pre_path[idx]
            )
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
            V_name = os.path.join(
                self.src_dir, 
                self.vowel_path[idx]
            )

            S_data, sample_rate_S = torchaudio.load(S_name, normalize=True)
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            V_data, sample_rate_V = torchaudio.load(V_name, normalize=True)
            assert sample_rate_S == sample_rate_T == sample_rate_V

            data = torch.cat([S_data, T_data, V_data], dim=1)
            phoneseq = torch.tensor([self.mapper.encode(segment) for segment in ['S', self.stop_name[idx], self.vowel_name[idx]]], 
                                    dtype=torch.long)
        else: 
            # "T"
            # NOTE: in this way we equate STV and #TV conditions and they are now directly comparable. 
            Sil_duration = self.silence_duration[idx]
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
            V_name = os.path.join(
                self.src_dir, 
                self.vowel_path[idx]
            )
            Sil_data = self.noise_set[idx]
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            V_data, sample_rate_V = torchaudio.load(V_name, normalize=True)
            assert sample_rate_T == sample_rate_V

            data = torch.cat([Sil_data, T_data, V_data], dim=1)
            phoneseq = torch.tensor([self.mapper.encode(segment) for segment in [self.stop_name[idx] + self.plosive_suffix, self.vowel_name[idx]]], 
                                    dtype=torch.long)   # 'SIL', 

        if self.transform:
            data = self.transform(data)
        
        return data, self.phi_type[idx], self.stop_name[idx], self.vowel_name[idx], self.first_sep_frame[idx], self.second_sep_frame[idx], phoneseq

    @staticmethod
    def collate_fn(data):
        # only working for one data at the moment
        batch_first = True
        xx, pt, sn, vn, sf1, sf2, phoneseq = zip(*data)
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return xx_pad, x_lens, pt, sn, vn, sf1, sf2, phoneseq
    
class TargetVowelDatasetBoundaryPhoneseqBothSIL(Dataset): 
    # this mixes T and ST. 
    # Target means the phenomenon-target, that is, e.g. /th/ or /st/. 
    # NOTE: for TV condition we add silence as # in the place of S
    # NOTE: this version does not generate random silence here, because for evaluation 
    # NOTE: silence can be generated outside. 
    # NOTE: For this version, we add silence in both ST and T conditions. 
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None, plosive_suffix=""):
        # guide_file = pd.read_csv(guide_)
        if isinstance(guide_, str):
            guide_file = pd.read_csv(guide_)
        elif isinstance(guide_, pd.DataFrame):
            guide_file = guide_
        else:
            raise Exception("Guide neither to read or to be used directly")
        
        self.plosive_suffix = plosive_suffix
        
        guide_file["zeroth_sep_frame"] = guide_file.apply(lambda x: time_to_frame(x['pre_startTime'] - x['sil_startTime']), axis=1)
        # 0-th sep-frame: st正常计算，t因为pre_startTime==stop_startTime，所以zeroth_sep_frame应等于first_sep_frame
        guide_file["first_sep_frame"] = guide_file.apply(lambda x: time_to_frame(x['stop_startTime'] - x['sil_startTime']), axis=1)
        guide_file["second_sep_frame"] = guide_file.apply(lambda x: time_to_frame(x['vowel_startTime'] - x['sil_startTime']), axis=1)
        
        pre_path_col = guide_file["pre_path"]
        stop_path_col = guide_file["stop_path"]
        vowel_path_col = guide_file["vowel_path"]
        phi_type_col = guide_file["phi_type"]
        stop_name_col = guide_file["stop"]
        vowel_name_col = guide_file["vowel"]
        zeroth_sep_frame_col = guide_file["zeroth_sep_frame"]
        first_sep_frame_col = guide_file["first_sep_frame"]
        second_sep_frame_col = guide_file["second_sep_frame"]
        
        # 这样一来，ST和T都这么计算
        self.silence_duration = (guide_file['pre_startTime'] - guide_file['sil_startTime']).tolist()

        self.guide_file = guide_file
        self.dataset = stop_path_col.tolist()
        self.pre_path = pre_path_col.tolist()
        self.vowel_path = vowel_path_col.tolist()
        self.phi_type = phi_type_col.tolist()
        self.stop_name = stop_name_col.tolist()
        self.vowel_name = vowel_name_col.tolist()
        self.zeroth_sep_frame = zeroth_sep_frame_col.tolist()
        self.first_sep_frame = first_sep_frame_col.tolist()
        self.second_sep_frame = second_sep_frame_col.tolist()
        self.src_dir = src_dir
        self.transform = transform

        self.mapper = mapper
        noise_gen = WhiteNoiseGen(sample_rate=16000, amplitude_scale=0.01)
        self.noise_set = noise_gen.generate_samples(np.array(self.silence_duration))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.phi_type[idx] == "ST": 
            # read two and concat
            S_name = os.path.join(
                self.src_dir, 
                self.pre_path[idx]
            )
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
            V_name = os.path.join(
                self.src_dir, 
                self.vowel_path[idx]
            )

            Sil_data = self.noise_set[idx]
            S_data, sample_rate_S = torchaudio.load(S_name, normalize=True)
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            V_data, sample_rate_V = torchaudio.load(V_name, normalize=True)
            assert sample_rate_S == sample_rate_T == sample_rate_V

            data = torch.cat([Sil_data, S_data, T_data, V_data], dim=1)
            phoneseq = torch.tensor([self.mapper.encode(segment) for segment in ['S', self.stop_name[idx], self.vowel_name[idx]]], 
                                    dtype=torch.long)
        else: 
            # "T"
            # NOTE: in this way we equate STV and #TV conditions and they are now directly comparable. 
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
            V_name = os.path.join(
                self.src_dir, 
                self.vowel_path[idx]
            )
            Sil_data = self.noise_set[idx]
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            V_data, sample_rate_V = torchaudio.load(V_name, normalize=True)
            assert sample_rate_T == sample_rate_V

            data = torch.cat([Sil_data, T_data, V_data], dim=1)
            phoneseq = torch.tensor([self.mapper.encode(segment) for segment in [self.stop_name[idx] + self.plosive_suffix, self.vowel_name[idx]]], 
                                    dtype=torch.long)   # 'SIL', 

        if self.transform:
            data = self.transform(data)
        
        return data, self.phi_type[idx], self.stop_name[idx], self.vowel_name[idx], self.zeroth_sep_frame[idx], self.first_sep_frame[idx], self.second_sep_frame[idx], phoneseq

    @staticmethod
    def collate_fn(data):
        # only working for one data at the moment
        batch_first = True
        xx, pt, sn, vn, sf0, sf1, sf2, phoneseq = zip(*data)
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return xx_pad, x_lens, pt, sn, vn, sf0, sf1, sf2, phoneseq

class TargetVowelDatasetBoundaryPhoneseqOnlyTV(Dataset): 
    # NOTE: This is to test whether it is at the start that makes STV and TV different.
    # this mixes T and ST. 
    # Target means the phenomenon-target, that is, e.g. /th/ or /st/. 
    # NOTE: for TV condition we add silence as # in the place of S
    # NOTE: this version does not generate random silence here, because for evaluation 
    # NOTE: silence can be generated outside. 
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None, plosive_suffix=""):
        # guide_file = pd.read_csv(guide_)
        if isinstance(guide_, str):
            guide_file = pd.read_csv(guide_)
        elif isinstance(guide_, pd.DataFrame):
            guide_file = guide_
        else:
            raise Exception("Guide neither to read or to be used directly")
        
        self.plosive_suffix = plosive_suffix
        
        guide_file["first_sep_frame"] = 0
        guide_file["second_sep_frame"] = guide_file.apply(lambda x: time_to_frame(x['vowel_startTime'] - x['stop_startTime']), axis=1)
        
        pre_path_col = guide_file["pre_path"]
        stop_path_col = guide_file["stop_path"]
        vowel_path_col = guide_file["vowel_path"]
        phi_type_col = guide_file["phi_type"]
        stop_name_col = guide_file["stop"]
        vowel_name_col = guide_file["vowel"]
        first_sep_frame_col = guide_file["first_sep_frame"]
        second_sep_frame_col = guide_file["second_sep_frame"]
        
        # generate random length silence for TV condition
        self.silence_duration = (guide_file['stop_startTime'] - guide_file['pre_startTime']).tolist()

        self.guide_file = guide_file
        self.dataset = stop_path_col.tolist()
        self.pre_path = pre_path_col.tolist()
        self.vowel_path = vowel_path_col.tolist()
        self.phi_type = phi_type_col.tolist()
        self.stop_name = stop_name_col.tolist()
        self.vowel_name = vowel_name_col.tolist()
        self.first_sep_frame = first_sep_frame_col.tolist()
        self.second_sep_frame = second_sep_frame_col.tolist()
        self.src_dir = src_dir
        self.transform = transform

        self.mapper = mapper
        noise_gen = WhiteNoiseGen(sample_rate=16000, amplitude_scale=0.01)
        self.noise_set = noise_gen.generate_samples(np.array(self.silence_duration))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.phi_type[idx] == "ST": 
            # read two and concat
            S_name = os.path.join(
                self.src_dir, 
                self.pre_path[idx]
            )
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
            V_name = os.path.join(
                self.src_dir, 
                self.vowel_path[idx]
            )

            # S_data, sample_rate_S = torchaudio.load(S_name, normalize=True)
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            V_data, sample_rate_V = torchaudio.load(V_name, normalize=True)
            assert sample_rate_T == sample_rate_V

            data = torch.cat([T_data, V_data], dim=1)
            phoneseq = torch.tensor([self.mapper.encode(segment) for segment in [self.stop_name[idx], self.vowel_name[idx]]], 
                                    dtype=torch.long)
        else: 
            # "T"
            # NOTE: in this way we equate STV and #TV conditions and they are now directly comparable. 
            Sil_duration = self.silence_duration[idx]
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
            V_name = os.path.join(
                self.src_dir, 
                self.vowel_path[idx]
            )
            Sil_data = self.noise_set[idx]
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            V_data, sample_rate_V = torchaudio.load(V_name, normalize=True)
            assert sample_rate_T == sample_rate_V

            data = torch.cat([T_data, V_data], dim=1)
            phoneseq = torch.tensor([self.mapper.encode(segment) for segment in [self.stop_name[idx] + self.plosive_suffix, self.vowel_name[idx]]], 
                                    dtype=torch.long)   # 'SIL', 

        if self.transform:
            data = self.transform(data)
        
        return data, self.phi_type[idx], self.stop_name[idx], self.vowel_name[idx], self.first_sep_frame[idx], self.second_sep_frame[idx], phoneseq

    @staticmethod
    def collate_fn(data):
        # only working for one data at the moment
        batch_first = True
        xx, pt, sn, vn, sf1, sf2, phoneseq = zip(*data)
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return xx_pad, x_lens, pt, sn, vn, sf1, sf2, phoneseq

class TargetVowelDatasetBoundaryPhoneseqNoSil(Dataset): 
    # NOTE: This is to test whether it is at the start that makes STV and TV different.
    # this mixes T and ST. 
    # Target means the phenomenon-target, that is, e.g. /th/ or /st/. 
    # NOTE: for TV condition we add silence as # in the place of S
    # NOTE: this version does not generate random silence here, because for evaluation 
    # NOTE: silence can be generated outside. 
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None, plosive_suffix=""):
        # guide_file = pd.read_csv(guide_)
        if isinstance(guide_, str):
            guide_file = pd.read_csv(guide_)
        elif isinstance(guide_, pd.DataFrame):
            guide_file = guide_
        else:
            raise Exception("Guide neither to read or to be used directly")
        
        self.plosive_suffix = plosive_suffix
        
        guide_file["first_sep_frame"] = guide_file.apply(lambda x: time_to_frame(x['stop_startTime'] - x['pre_startTime']), axis=1)
        guide_file["second_sep_frame"] = guide_file.apply(lambda x: time_to_frame(x['vowel_startTime'] - x['pre_startTime']), axis=1)
        
        pre_path_col = guide_file["pre_path"]
        stop_path_col = guide_file["stop_path"]
        vowel_path_col = guide_file["vowel_path"]
        phi_type_col = guide_file["phi_type"]
        stop_name_col = guide_file["stop"]
        vowel_name_col = guide_file["vowel"]
        first_sep_frame_col = guide_file["first_sep_frame"]
        second_sep_frame_col = guide_file["second_sep_frame"]
        
        # generate random length silence for TV condition
        self.silence_duration = (guide_file['stop_startTime'] - guide_file['pre_startTime']).tolist()

        self.guide_file = guide_file
        self.dataset = stop_path_col.tolist()
        self.pre_path = pre_path_col.tolist()
        self.vowel_path = vowel_path_col.tolist()
        self.phi_type = phi_type_col.tolist()
        self.stop_name = stop_name_col.tolist()
        self.vowel_name = vowel_name_col.tolist()
        self.first_sep_frame = first_sep_frame_col.tolist()
        self.second_sep_frame = second_sep_frame_col.tolist()
        self.src_dir = src_dir
        self.transform = transform

        self.mapper = mapper
        noise_gen = WhiteNoiseGen(sample_rate=16000, amplitude_scale=0.01)
        self.noise_set = noise_gen.generate_samples(np.array(self.silence_duration))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.phi_type[idx] == "ST": 
            # read two and concat
            S_name = os.path.join(
                self.src_dir, 
                self.pre_path[idx]
            )
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
            V_name = os.path.join(
                self.src_dir, 
                self.vowel_path[idx]
            )

            # S_data, sample_rate_S = torchaudio.load(S_name, normalize=True)
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            V_data, sample_rate_V = torchaudio.load(V_name, normalize=True)
            assert sample_rate_T == sample_rate_V

            data = torch.cat([T_data, V_data], dim=1)
            phoneseq = torch.tensor([self.mapper.encode(segment) for segment in [self.stop_name[idx], self.vowel_name[idx]]], 
                                    dtype=torch.long)
        else: 
            # "T"
            # NOTE: in this way we equate STV and #TV conditions and they are now directly comparable. 
            Sil_duration = self.silence_duration[idx]
            T_name = os.path.join(
                self.src_dir, 
                self.dataset[idx]
            )
            V_name = os.path.join(
                self.src_dir, 
                self.vowel_path[idx]
            )
            Sil_data = self.noise_set[idx]
            T_data, sample_rate_T = torchaudio.load(T_name, normalize=True)
            V_data, sample_rate_V = torchaudio.load(V_name, normalize=True)
            assert sample_rate_T == sample_rate_V

            data = torch.cat([T_data, V_data], dim=1)
            phoneseq = torch.tensor([self.mapper.encode(segment) for segment in [self.stop_name[idx] + self.plosive_suffix, self.vowel_name[idx]]], 
                                    dtype=torch.long)   # 'SIL', 

        if self.transform:
            data = self.transform(data)
        
        return data, self.phi_type[idx], self.stop_name[idx], self.vowel_name[idx], self.first_sep_frame[idx], self.second_sep_frame[idx], phoneseq

    @staticmethod
    def collate_fn(data):
        # only working for one data at the moment
        batch_first = True
        xx, pt, sn, vn, sf1, sf2, phoneseq = zip(*data)
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return xx_pad, x_lens, pt, sn, vn, sf1, sf2, phoneseq























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


class MelSpecTransformDB(nn.Module): 
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
        # n_stft = int((n_fft//2) + 1)
        n_stft = n_fft//2 + 1
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
    def norm_minmax(mel_spec, non_normed_mel_spec):
        min_val = non_normed_mel_spec.min()
        max_val = non_normed_mel_spec.max()
        mel_spec = mel_spec * (max_val - min_val) + min_val
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