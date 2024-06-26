import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import random
from IPython.display import Audio
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from collections import Counter
from sklearn.manifold import TSNE   # one type of clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from itertools import combinations
from sklearn.decomposition import PCA
from scipy.linalg import block_diag
import pickle
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats import sem
import argparse

from model_padding import generate_mask_from_lengths_mat, mask_it
from paths import *
from misc_my_utils import *
from model_loss import *
from model_dataset import Normalizer, DeNormalizer, TokenMap
from model_dataset import MelSpecTransformDB as TheTransform
from model_dataset import DS_Tools
from reshandler import DictResHandler
from misc_progress_bar import draw_progress_bar
from test_bnd_detect_tools import *
from misc_tools import PathUtils as PU
from misc_tools import AudioCut, ARPABET
from misc_my_utils import time_to_frame

# Constants
EPOCHS = 10
BATCH_SIZE = 1
INPUT_DIM = 64
OUTPUT_DIM = 64
INTER_DIM_0 = 32
INTER_DIM_1 = 16
INTER_DIM_2 = 8
ENC_SIZE_LIST = [INPUT_DIM, INTER_DIM_0, INTER_DIM_1, INTER_DIM_2]
DEC_SIZE_LIST = [OUTPUT_DIM, INTER_DIM_0, INTER_DIM_1, INTER_DIM_2]
DROPOUT = 0.5
NUM_LAYERS = 2
EMBEDDING_DIM = 128
REC_SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 64
LOADER_WORKER = 16

############################################ Utils ############################################
def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    # ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
    ax.imshow(specgram, origin="lower", aspect="auto", interpolation="nearest")

def get_endframes(seppos, attn_size): 
    return [0, seppos], [seppos, attn_size]

def create_phoneme_block_matrix(starts, ends, total):
    # Initialize an empty list to store phoneme block matrices
    phoneme_blocks = []
    # Iterate through the phoneme frames to create each block
    for start_frame, end_frame in list(zip(starts, ends))[:-1]:
        num_frames = end_frame - start_frame
        phoneme_block = np.ones((num_frames, num_frames))
        phoneme_blocks.append(phoneme_block)
    num_frames = total - starts[-1]
    phoneme_block = np.ones((num_frames, num_frames))
    phoneme_blocks.append(phoneme_block)
    block_diag_matrix = block_diag(*phoneme_blocks)
    return block_diag_matrix

def post2pre_filter(start, sep, end): 
    return np.block([[np.zeros((sep-start, end))], [np.ones((end-sep, sep-start)), np.zeros((end-sep, end-sep))]])

def biway_filter(start, sep, end): 
    return np.block([[np.zeros((sep-start, sep-start)), np.ones((sep-start, end-sep))], [np.ones((end-sep, sep-start)), np.zeros((end-sep, end-sep))]])

def get_in_phone_attn(attn, starts, ends, total): 
    block_diag_matrix = create_phoneme_block_matrix(starts, ends, total)
    filtered_attn = block_diag_matrix * attn
    in_phoneme_attn = filtered_attn.sum(-1)
    return in_phoneme_attn

def interpolate_traj(current, n_steps=100): 
    current_steps = np.linspace(0, 1, num=len(current))
    target_steps = np.linspace(0, 1, num=n_steps)
    interp_func = interp1d(current_steps, current, kind='linear', fill_value="extrapolate")
    return interp_func(target_steps)

def cutHid(hid, cutstart, cutend, start_offset=0, end_offset=1): 
    selstart = max(cutstart, int(cutstart + (cutend - cutstart) * start_offset))
    selend = min(cutend, int(cutstart + (cutend - cutstart) * end_offset))
    # hid is (L, H)
    return hid[selstart:selend, :]

# we have very limited data, so we don't need to select, just plot all
def get_toplot(hiddens, sepframes, phi_types, stop_names, offsets=(0, 1)): 
    cutstarts = []
    cutends = []
    for hidden, sepframe, phi_type in zip(hiddens, sepframes, phi_types):
        if phi_type == 'ST':
            cutstarts.append(sepframe)
        else:
            cutstarts.append(0)
        cutends.append(hidden.shape[0])
    
    hid_sel = np.empty((0, 8))
    tag_sel = []
    for (item, start, end, tag) in zip(hiddens, cutstarts, cutends, phi_types): 
        hid = cutHid(item, start, end, offsets[0], offsets[1])
        hidlen = hid.shape[0]
        hid_sel = np.concatenate((hid_sel, hid), axis=0)
        tag_sel += [tag] * hidlen
    return hid_sel, np.array(tag_sel)

def extract_attention_blocks_ST(attention_matrix, sepframe1, sepframe2):
    t_start, t_end = sepframe1, sepframe2
    s_start, s_end = 0, sepframe1
    a_start, a_end = sepframe2, attention_matrix.shape[0]
    # (start, end] for slicing
    # Extracting specific blocks based on the provided indices
    t_to_s = attention_matrix[t_start:t_end, s_start:s_end].sum(axis=1)
    s_to_t = attention_matrix[s_start:s_end, t_start:t_end].sum(axis=1)
    a_to_t = attention_matrix[a_start:a_end, t_start:t_end].sum(axis=1)
    t_to_a = attention_matrix[t_start:t_end, a_start:a_end].sum(axis=1)
    
    return {
        't_to_s': t_to_s,
        's_to_t': s_to_t,
        'a_to_t': a_to_t,
        't_to_a': t_to_a
    }

def extract_attention_blocks_T(attention_matrix, sepframe):
    t_start, t_end = 0, sepframe
    a_start, a_end = sepframe, attention_matrix.shape[0]
    # (start, end] for slicing
    # Extracting specific blocks based on the provided indices
    t_to_a = attention_matrix[t_start:t_end, a_start:a_end].sum(axis=1)
    a_to_t = attention_matrix[a_start:a_end, t_start:t_end].sum(axis=1)
    
    return {
        't_to_a': t_to_a,
        'a_to_t': a_to_t
    }

def plot_attention_trajectory(phi_type, all_attn, all_sepframes1, all_sepframes2, save_path): 
    if phi_type == "ST":
        n_steps = 100
        s_to_t_traj = []
        t_to_s_traj = []
        t_to_a_traj = []
        a_to_t_traj = []
        for i in range(len(all_attn)): 
            this_attn = all_attn[i]
            this_sep_frame1 = all_sepframes1[i]
            this_sep_frame2 = all_sepframes2[i]

            blocks = extract_attention_blocks_ST(this_attn, this_sep_frame1, this_sep_frame2)

            s_to_t_interp = interpolate_traj(blocks['s_to_t'], n_steps)
            t_to_s_interp = interpolate_traj(blocks['t_to_s'], n_steps)
            t_to_a_interp = interpolate_traj(blocks['t_to_a'], n_steps)
            a_to_t_interp = interpolate_traj(blocks['a_to_t'], n_steps)
            s_to_t_traj.append(s_to_t_interp)
            t_to_s_traj.append(t_to_s_interp)
            t_to_a_traj.append(t_to_a_interp)
            a_to_t_traj.append(a_to_t_interp)

        # Convert list of arrays into 2D NumPy arrays for easier manipulation
        group1_array = np.array(s_to_t_traj)
        group2_array = np.array(t_to_s_traj)
        group3_array = np.array(t_to_a_traj)
        group4_array = np.array(a_to_t_traj)

        # Calculate the mean trajectory for each group
        means = np.array([np.mean(group1_array, axis=0), 
                        np.mean(group2_array, axis=0), 
                        np.mean(group3_array, axis=0), 
                        np.mean(group4_array, axis=0)])

        # Calculate the SEM for each step in both groups
        sems = np.array([sem(group1_array, axis=0),
                        sem(group2_array, axis=0),
                        sem(group3_array, axis=0),
                        sem(group4_array, axis=0)])

        # Calculate the 95% CI for both groups
        ci_95s = 1.96 * sems

        # Upper and lower bounds of the 95% CI for both groups
        upper_bounds = means + ci_95s
        lower_bounds = means - ci_95s

        # Plotting
        plt.figure(figsize=(12, 8))
        for mean, upper, lower, label in zip(means, upper_bounds, lower_bounds, ['S-to-P', 'P-to-S', 'P-to-V', 'V-to-P']):
            plt.plot(mean, label=label)
            plt.fill_between(range(n_steps), lower, upper, alpha=0.2)

        plt.xlabel('Normalized Time')
        plt.ylabel('Summed Foreign-Attention')
        plt.title('Comparison of Foreign-Attention Trajectory')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
    elif phi_type == "T": 
        n_steps = 100
        t_to_a_traj = []
        a_to_t_traj = []
        for i in range(len(all_attn)): 
            this_attn = all_attn[i]
            this_sep_frame2 = all_sepframes2[i]

            blocks = extract_attention_blocks_T(this_attn, this_sep_frame2)

            t_to_a_interp = interpolate_traj(blocks['t_to_a'], n_steps)
            a_to_t_interp = interpolate_traj(blocks['a_to_t'], n_steps)
            t_to_a_traj.append(t_to_a_interp)
            a_to_t_traj.append(a_to_t_interp)

        # Convert list of arrays into 2D NumPy arrays for easier manipulation
        group3_array = np.array(t_to_a_traj)
        group4_array = np.array(a_to_t_traj)

        # Calculate the mean trajectory for each group
        means = np.array([np.mean(group3_array, axis=0), 
                        np.mean(group4_array, axis=0)])

        # Calculate the SEM for each step in both groups
        sems = np.array([sem(group3_array, axis=0),
                        sem(group4_array, axis=0)])

        # Calculate the 95% CI for both groups
        ci_95s = 1.96 * sems

        # Upper and lower bounds of the 95% CI for both groups
        upper_bounds = means + ci_95s
        lower_bounds = means - ci_95s

        # Plotting
        plt.figure(figsize=(12, 8))
        for mean, upper, lower, label in zip(means, upper_bounds, lower_bounds, ['P-to-V', 'V-to-P']):
            plt.plot(mean, label=label)
            plt.fill_between(range(n_steps), lower, upper, alpha=0.2)

        plt.xlabel('Normalized Time')
        plt.ylabel('Summed Foreign-Attention')
        plt.title('Comparison of Foreign-Attention Trajectory')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

def plot_attention_trajectory_together(all_phi_type, all_attn, all_sepframes1, all_sepframes2, save_path): 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    legend_namess = [['S-to-P', 'P-to-S', 'P-to-V', 'V-to-P'], ['#-to-P', 'P-to-#', 'P-to-V', 'V-to-P']]
    colors = ['b', 'g', 'red', 'orange']
    n_steps = 100

    for (selector, ax, legend_names) in zip(["ST", "T"], [ax1, ax2], legend_namess):
        selected_tuples = [(sf1, sf2, attn) for pt, sf1, sf2, attn in zip(all_phi_type,  
                                                          all_sepframes1, 
                                                          all_sepframes2, 
                                                          all_attn) if pt == selector]
        selected_sf1s, selected_sf2s, selected_attns = zip(*selected_tuples)
        if selector == "ST":
            s_to_t_traj = []
            t_to_s_traj = []
            t_to_a_traj = []
            a_to_t_traj = []
            for i in range(len(selected_attns)): 
                this_attn = selected_attns[i]
                this_sep_frame1 = selected_sf1s[i]
                this_sep_frame2 = selected_sf2s[i]

                blocks = extract_attention_blocks_ST(this_attn, this_sep_frame1, this_sep_frame2)

                s_to_t_interp = interpolate_traj(blocks['s_to_t'], n_steps)
                t_to_s_interp = interpolate_traj(blocks['t_to_s'], n_steps)
                t_to_a_interp = interpolate_traj(blocks['t_to_a'], n_steps)
                a_to_t_interp = interpolate_traj(blocks['a_to_t'], n_steps)
                s_to_t_traj.append(s_to_t_interp)
                t_to_s_traj.append(t_to_s_interp)
                t_to_a_traj.append(t_to_a_interp)
                a_to_t_traj.append(a_to_t_interp)

            # Convert list of arrays into 2D NumPy arrays for easier manipulation
            group1_array = np.array(s_to_t_traj)
            group2_array = np.array(t_to_s_traj)
            group3_array = np.array(t_to_a_traj)
            group4_array = np.array(a_to_t_traj)

            # Calculate the mean trajectory for each group
            means = np.array([np.mean(group1_array, axis=0), 
                            np.mean(group2_array, axis=0), 
                            np.mean(group3_array, axis=0), 
                            np.mean(group4_array, axis=0)])

            # Calculate the SEM for each step in both groups
            sems = np.array([sem(group1_array, axis=0),
                            sem(group2_array, axis=0),
                            sem(group3_array, axis=0),
                            sem(group4_array, axis=0)])

            # Calculate the 95% CI for both groups
            ci_95s = 1.96 * sems

            # Upper and lower bounds of the 95% CI for both groups
            upper_bounds = means + ci_95s
            lower_bounds = means - ci_95s

            for mean, upper, lower, label, c in zip(means, upper_bounds, lower_bounds, legend_names, colors):
                ax.plot(mean, label=label, color=c)
                ax.fill_between(range(n_steps), lower, upper, alpha=0.2, color=c)

        elif selector == "T": 
            s_to_t_traj = []
            t_to_s_traj = []
            t_to_a_traj = []
            a_to_t_traj = []
            for i in range(len(selected_attns)): 
                this_attn = selected_attns[i]
                this_sep_frame1 = selected_sf1s[i]
                this_sep_frame2 = selected_sf2s[i]

                blocks = extract_attention_blocks_ST(this_attn, this_sep_frame1, this_sep_frame2)

                s_to_t_interp = interpolate_traj(blocks['s_to_t'], n_steps)
                t_to_s_interp = interpolate_traj(blocks['t_to_s'], n_steps)
                t_to_a_interp = interpolate_traj(blocks['t_to_a'], n_steps)
                a_to_t_interp = interpolate_traj(blocks['a_to_t'], n_steps)
                s_to_t_traj.append(s_to_t_interp)
                t_to_s_traj.append(t_to_s_interp)
                t_to_a_traj.append(t_to_a_interp)
                a_to_t_traj.append(a_to_t_interp)

            # Convert list of arrays into 2D NumPy arrays for easier manipulation
            group1_array = np.array(s_to_t_traj)
            group2_array = np.array(t_to_s_traj)
            group3_array = np.array(t_to_a_traj)
            group4_array = np.array(a_to_t_traj)

            # Calculate the mean trajectory for each group
            means = np.array([np.mean(group1_array, axis=0), 
                            np.mean(group2_array, axis=0), 
                            np.mean(group3_array, axis=0), 
                            np.mean(group4_array, axis=0)])

            # Calculate the SEM for each step in both groups
            sems = np.array([sem(group1_array, axis=0),
                            sem(group2_array, axis=0),
                            sem(group3_array, axis=0),
                            sem(group4_array, axis=0)])

            # Calculate the 95% CI for both groups
            ci_95s = 1.96 * sems

            # Upper and lower bounds of the 95% CI for both groups
            upper_bounds = means + ci_95s
            lower_bounds = means - ci_95s

            for mean, upper, lower, label, c in zip(means, upper_bounds, lower_bounds, legend_names, colors):
                ax.plot(mean, label=label, color=c)
                ax.fill_between(range(n_steps), lower, upper, alpha=0.2, color=c)
        else: 
            t_to_a_traj = []
            a_to_t_traj = []
            for i in range(len(selected_attns)): 
                this_attn = selected_attns[i]
                this_sep_frame2 = selected_sf2s[i]

                blocks = extract_attention_blocks_T(this_attn, this_sep_frame2)

                t_to_a_interp = interpolate_traj(blocks['t_to_a'], n_steps)
                a_to_t_interp = interpolate_traj(blocks['a_to_t'], n_steps)
                t_to_a_traj.append(t_to_a_interp)
                a_to_t_traj.append(a_to_t_interp)

            # Convert list of arrays into 2D NumPy arrays for easier manipulation
            group3_array = np.array(t_to_a_traj)
            group4_array = np.array(a_to_t_traj)

            # Calculate the mean trajectory for each group
            means = np.array([np.mean(group3_array, axis=0), 
                            np.mean(group4_array, axis=0)])

            # Calculate the SEM for each step in both groups
            sems = np.array([sem(group3_array, axis=0),
                            sem(group4_array, axis=0)])

            # Calculate the 95% CI for both groups
            ci_95s = 1.96 * sems

            # Upper and lower bounds of the 95% CI for both groups
            upper_bounds = means + ci_95s
            lower_bounds = means - ci_95s

            for mean, upper, lower, label, c in zip(means, upper_bounds, lower_bounds, legend_names[2:], colors[2:]):
                ax.plot(mean, label=label, color=c)
                ax.fill_between(range(n_steps), lower, upper, alpha=0.2, color=c)
        ax.set_xlabel('Normalized Time')
        ax.set_ylabel('Summed Foreign-Attention')
        ax.set_title(f'{selector}')
        ax.set_ylim([0, 1])
        ax.legend(loc = "upper left")
        ax.grid(True)

    fig.suptitle('Comparison of Foreign-Attention Trajectory')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def extract_attention_blocks_ST012(attention_matrix, sepframe0, sepframe1, sepframe2):
    t_start, t_end = sepframe1, sepframe2
    s_start, s_end = sepframe0, sepframe1
    a_start, a_end = sepframe2, attention_matrix.shape[0]
    # (start, end] for slicing
    # Extracting specific blocks based on the provided indices
    t_to_s = attention_matrix[t_start:t_end, s_start:s_end].sum(axis=1)
    s_to_t = attention_matrix[s_start:s_end, t_start:t_end].sum(axis=1)
    a_to_t = attention_matrix[a_start:a_end, t_start:t_end].sum(axis=1)
    t_to_a = attention_matrix[t_start:t_end, a_start:a_end].sum(axis=1)
    
    return {
        't_to_s': t_to_s,
        's_to_t': s_to_t,
        'a_to_t': a_to_t,
        't_to_a': t_to_a
    }

def extract_attention_blocks_T012(attention_matrix, sepframe0, sepframe1, sepframe2): 
    # TV中sepframe0应与sepframe1相同
    t_start, t_end = sepframe1, sepframe2
    s_start, s_end = 0, sepframe1
    a_start, a_end = sepframe2, attention_matrix.shape[0]
    # (start, end] for slicing
    # Extracting specific blocks based on the provided indices
    t_to_s = attention_matrix[t_start:t_end, s_start:s_end].sum(axis=1)
    s_to_t = attention_matrix[s_start:s_end, t_start:t_end].sum(axis=1)
    a_to_t = attention_matrix[a_start:a_end, t_start:t_end].sum(axis=1)
    t_to_a = attention_matrix[t_start:t_end, a_start:a_end].sum(axis=1)
    
    return {
        't_to_s': t_to_s,
        's_to_t': s_to_t,
        'a_to_t': a_to_t,
        't_to_a': t_to_a
    }

def plot_attention_trajectory_together_012(all_phi_type, all_attn, all_sepframes0, all_sepframes1, all_sepframes2, save_path): 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    legend_namess = [['S-to-P', 'P-to-S', 'P-to-V', 'V-to-P'], ['#-to-P', 'P-to-#', 'P-to-V', 'V-to-P']]
    colors = ['b', 'g', 'red', 'orange']
    n_steps = 100
    badcounts = {"ST": 0, "T": 0}
    totalcounts = {"ST": 0, "T": 0}

    for (selector, ax, legend_names) in zip(["ST", "T"], [ax1, ax2], legend_namess):
        selected_tuples = [(sf0, sf1, sf2, attn) for pt, sf0, sf1, sf2, attn in zip(all_phi_type,  
                                                          all_sepframes0, 
                                                          all_sepframes1, 
                                                          all_sepframes2, 
                                                          all_attn) if pt == selector]
        selected_sf0s, selected_sf1s, selected_sf2s, selected_attns = zip(*selected_tuples)
        if selector == "ST":
            s_to_t_traj = []
            t_to_s_traj = []
            t_to_a_traj = []
            a_to_t_traj = []
            totalcounts["ST"] += len(selected_attns)
            for i in range(len(selected_attns)): 
                this_attn = selected_attns[i]
                this_sep_frame0 = selected_sf0s[i]
                this_sep_frame1 = selected_sf1s[i]
                this_sep_frame2 = selected_sf2s[i]

                blocks = extract_attention_blocks_ST012(this_attn, this_sep_frame0, this_sep_frame1, this_sep_frame2)

                s_to_t_interp = interpolate_traj(blocks['s_to_t'], n_steps)
                t_to_s_interp = interpolate_traj(blocks['t_to_s'], n_steps)
                t_to_a_interp = interpolate_traj(blocks['t_to_a'], n_steps)
                a_to_t_interp = interpolate_traj(blocks['a_to_t'], n_steps)
                if np.any(np.isnan(s_to_t_interp)) or np.any(np.isnan(t_to_s_interp)) or np.any(np.isnan(t_to_a_interp)) or np.any(np.isnan(a_to_t_interp)):
                    badcounts['ST'] += 1
                    continue
                s_to_t_traj.append(s_to_t_interp)
                t_to_s_traj.append(t_to_s_interp)
                t_to_a_traj.append(t_to_a_interp)
                a_to_t_traj.append(a_to_t_interp)

            # Convert list of arrays into 2D NumPy arrays for easier manipulation
            group1_array = np.array(s_to_t_traj)
            group2_array = np.array(t_to_s_traj)
            group3_array = np.array(t_to_a_traj)
            group4_array = np.array(a_to_t_traj)

            # Calculate the mean trajectory for each group
            means = np.array([np.mean(group1_array, axis=0), 
                            np.mean(group2_array, axis=0), 
                            np.mean(group3_array, axis=0), 
                            np.mean(group4_array, axis=0)])

            # Calculate the SEM for each step in both groups
            sems = np.array([sem(group1_array, axis=0),
                            sem(group2_array, axis=0),
                            sem(group3_array, axis=0),
                            sem(group4_array, axis=0)])

            # Calculate the 95% CI for both groups
            ci_95s = 1.96 * sems

            # Upper and lower bounds of the 95% CI for both groups
            upper_bounds = means + ci_95s
            lower_bounds = means - ci_95s

            for mean, upper, lower, label, c in zip(means, upper_bounds, lower_bounds, legend_names, colors):
                ax.plot(mean, label=label, color=c)
                ax.fill_between(range(n_steps), lower, upper, alpha=0.2, color=c)

        elif selector == "T": 
            s_to_t_traj = []
            t_to_s_traj = []
            t_to_a_traj = []
            a_to_t_traj = []
            totalcounts["T"] += len(selected_attns)
            for i in range(len(selected_attns)): 
                this_attn = selected_attns[i]
                this_sep_frame0 = selected_sf0s[i]
                this_sep_frame1 = selected_sf1s[i]
                this_sep_frame2 = selected_sf2s[i]

                blocks = extract_attention_blocks_T012(this_attn, this_sep_frame0, this_sep_frame1, this_sep_frame2)

                s_to_t_interp = interpolate_traj(blocks['s_to_t'], n_steps)
                t_to_s_interp = interpolate_traj(blocks['t_to_s'], n_steps)
                t_to_a_interp = interpolate_traj(blocks['t_to_a'], n_steps)
                a_to_t_interp = interpolate_traj(blocks['a_to_t'], n_steps)
                if np.any(np.isnan(s_to_t_interp)) or np.any(np.isnan(t_to_s_interp)) or np.any(np.isnan(t_to_a_interp)) or np.any(np.isnan(a_to_t_interp)):
                    badcounts['T'] += 1
                    continue
                s_to_t_traj.append(s_to_t_interp)
                t_to_s_traj.append(t_to_s_interp)
                t_to_a_traj.append(t_to_a_interp)
                a_to_t_traj.append(a_to_t_interp)

            # Convert list of arrays into 2D NumPy arrays for easier manipulation
            group1_array = np.array(s_to_t_traj)
            group2_array = np.array(t_to_s_traj)
            group3_array = np.array(t_to_a_traj)
            group4_array = np.array(a_to_t_traj)

            # Calculate the mean trajectory for each group
            means = np.array([np.mean(group1_array, axis=0), 
                            np.mean(group2_array, axis=0), 
                            np.mean(group3_array, axis=0), 
                            np.mean(group4_array, axis=0)])

            # Calculate the SEM for each step in both groups
            sems = np.array([sem(group1_array, axis=0),
                            sem(group2_array, axis=0),
                            sem(group3_array, axis=0),
                            sem(group4_array, axis=0)])

            # Calculate the 95% CI for both groups
            ci_95s = 1.96 * sems

            # Upper and lower bounds of the 95% CI for both groups
            upper_bounds = means + ci_95s
            lower_bounds = means - ci_95s

            for mean, upper, lower, label, c in zip(means, upper_bounds, lower_bounds, legend_names, colors):
                ax.plot(mean, label=label, color=c)
                ax.fill_between(range(n_steps), lower, upper, alpha=0.2, color=c)
        else: 
            raise ValueError("Invalid selector")
        ax.set_xlabel('Normalized Time')
        ax.set_ylabel('Summed Foreign-Attention')
        ax.set_title(f'{selector}')
        ax.set_ylim([0, 1])
        ax.legend(loc = "upper left")
        ax.grid(True)

    print(f"badcounts: {badcounts}")
    print(f"totalcounts: {totalcounts}")
    fig.suptitle('Comparison of Foreign-Attention Trajectory')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
