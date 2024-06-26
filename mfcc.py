from python_speech_features import mfcc, delta
import numpy as np
import scipy.io.wavfile as wav
import pandas as pd

class MFCC_Processor: 
    def __init__(self): 
        # Following defaults in python_speech_features
        self.winlen = 0.025 # 25 ms
        self.winstep = 0.01 # 10 ms
        self.rate = 16000
    
    def time2frame(self, time_sec): 
        # Calculate the number of elapsed samples
        elapsed_samples = int(round(time_sec * self.rate))

        # Calculate the number of samples between adjacent frames
        frame_step = int(round(self.winstep * self.rate))

        # Calculate the frame index
        frame_index = int(round((elapsed_samples - self.winlen * self.rate) / frame_step))

        return frame_index
    
    def get_rec_len(self, wavepath): 
        """
        Get the length of the recording in seconds

        Args:
            wavepath (str): Path to the .wav file to be processed.

        Returns:
            length (float): Length of the recording in seconds.
        """
        # Read in the WAV file and its sampling rate
        (rate, data) = wav.read(wavepath)
        length = len(data) / rate
        return length

    def extract(self, wavepath):
        """
        Extracts Mel Frequency Cepstral Coefficients (MFCC) and its first and second order deltas from a single wave file. 
        MFCCs are a representation of the short-term power spectrum of a sound and are commonly used
        in speech recognition and music genre classification tasks. 

        Args:
            wavepath (str): Path to the .wav file to be processed.

        Returns:
            sound_feat (np.array): A 2D numpy array of shape (S, 39), where S is the number of MFCC samples and 39 is the total number of features extracted (13 MFCCs + 13 deltas + 13 delta-deltas).
        """
        # Read in the WAV file and its sampling rate
        (rate, sig) = wav.read(wavepath)
        self.rate = rate

        # Extract MFCCs
        mfcc_feat = mfcc(sig, samplerate=rate, winlen=self.winlen, winstep=self.winstep)

        # Compute deltas and delta-deltas of MFCCs
        delta_feat = delta(mfcc_feat, 2)
        delta_delta_feat = delta(delta_feat, 2)

        # Concatenate MFCCs, deltas and delta-deltas into a single feature vector
        sound_feat = np.concatenate((mfcc_feat, delta_feat, delta_delta_feat), axis=1)

        return sound_feat

    def cut_to_phones(self, mfcc, flat_starts, flat_ends): 
        flat_start_frames = [self.time2frame(t) for t in flat_starts]
        flat_end_frames = [self.time2frame(t) for t in flat_ends]
        
        phone_list = [mfcc[start:end] for start, end in zip(flat_start_frames, flat_end_frames)]
        
        return phone_list
    
    def group_phones(self, phone_list, cut_idx_starts, cut_idx_ends): 
        """
        Groups the phone numbers in the `phone_list` into sub-lists based on the indices defined by the `cut_idx_starts`and `cut_idx_ends` lists. The `cut_idx_starts` and `cut_idx_ends` lists define the start and end indices for each sub-list, respectively. The function returns a list of sub-lists, where each sub-list contains the phone numbers that fall within the range of the corresponding `cut_idx_starts` and `cut_idx_ends` values.
        It is also applicable to the grouping of corresponding ground truth tags. 

        Args:
        - self: a reference to the current instance of the class.
        - phone_list: a list of phone numbers.
        - cut_idx_starts: a list of start indices for the sub-lists.
        - cut_idx_ends: a list of end indices for the sub-lists.

        Returns:
        - A list of sub-lists of phone numbers, where each sub-list contains the phone numbers that fall within the 
          range of the corresponding `cut_idx_starts` and `cut_idx_ends` values.
        """
        return [phone_list[i:j] for i, j in zip(cut_idx_starts, cut_idx_ends)]
    
    def get_se(self, start_times, end_times, cut_idx_starts, cut_idx_ends): 
        cut_start_times = []
        cut_end_times = []
        for i, j in zip(cut_idx_starts, cut_idx_ends): 
            cut_start_times.append(start_times[i])
            cut_end_times.append(end_times[j - 1])
        return cut_start_times, cut_end_times
    
    def rebind_groups(self, groups): 
        """
        Rebinds the list of numpy arrays `groups` into a list of numpy arrays, where each numpy array is created by concatenating the arrays in the corresponding sub-list of `groups`. The function returns the list of rebinded numpy arrays.
    
    Args:
    - self: a reference to the current instance of the class.
    - groups: a list of numpy arrays sub-lists.
    
    Returns:
    - A list of rebinded numpy arrays, where each numpy array is created by concatenating the arrays in the corresponding sub-list of `groups`.
    """
        rebind_list = []
        for group in groups: 
            rebind = np.concatenate(group)
            rebind_list.append(rebind)
        return rebind_list
    
    def filter_len_zero_at(list_of_arrays, axis=0): 
        filtered_arrays = [arr for arr in list_of_arrays if arr.shape[axis] != 0]
        return filtered_arrays