# --------------------------
# Imports
# --------------------------
import os
import pickle
import pandas as pd
from paths import *
import torchaudio
import tempfile

# --------------------------
# Gamma Distribution
# --------------------------
def save_gamma_params(shape, loc, scale, path): 
    # Save the dictionary to a file using pickle
    params = {'shape': shape, 'loc': loc, 'scale': scale}
    with open(os.path.join(stat_params_path, path), 'wb') as f:
        pickle.dump(params, f)

def load_gamma_params(path): 
    # Open the file containing the saved dictionary
    with open(os.path.join(stat_params_path, path), 'rb') as file:
        # Load the dictionary from the file
        params = pickle.load(file)
    # Extract the parameters from the dictionary
    shape = params['shape']
    loc = params['loc']
    scale = params['scale']
    return shape, loc, scale




# --------------------------
# Token Read
# --------------------------

def filter_tokens_and_get_df(csv_path, keepSIL=False):
    df = pd.read_csv(csv_path)
    
    # Filter out tokens surrounded by <> and/or totally capitalized (but SIL might be special)
    if keepSIL: 
        filtered_df = df[~(df['token'].str.contains('<|>') | (df['token'] == df['token'].str.upper())) | (df['token'] == 'SIL')]
    else: 
        filtered_df = df[~(df['token'].str.contains('<|>') | (df['token'] == df['token'].str.upper()))]

    # Filter out rows where duration is equal to 0
    filtered_df = filtered_df[filtered_df['duration'] >= 0.0125]
    
    return filtered_df

# We filter out tokens surrounded by <> and all-capital tokens, as they are not word tokens.
# We also filter out tokens with duration equal to 0, as they are invalid.
def filter_tokens_and_get_durations(csv_path):
    filtered_df = filter_tokens_and_get_df(csv_path)
    
    # Get the durations of the filtered tokens and exclude those with a duration of 0
    durations = filtered_df['duration'].to_numpy()
    
    return durations

# --------------------------
# Cut & CSV IO
# --------------------------
def save_cut_waves_and_log(save_dir, log_dir, cut_list, corr_df): 
    rec_name = corr_df["rec"][0]
    suffices = corr_df["idx"]
    for cut_idx in range(len(cut_list)): 
        cut_seg = cut_list[cut_idx]
        save_name = os.path.join(save_dir, "{}_{}.wav".format(rec_name, suffices[cut_idx]))
        with tempfile.TemporaryDirectory() as tempdir:
            torchaudio.save(save_name, cut_seg, sample_rate=16000, encoding="PCM_S", bits_per_sample=16)  # fixed sample_rate, attention! 
    
    csv_path = os.path.join(log_dir, "{}.csv".format(rec_name))
    # write the combined DataFrame to the same CSV file
    corr_df.to_csv(csv_path, index=False)

# --------------------------
# Cut IO
# --------------------------
def save_cut(cut_list, save_dir, rec_name): 
    with open(os.path.join(save_dir, "{}.cut".format(rec_name)), 'wb') as f:
        pickle.dump(cut_list, f)

def load_cut(cut_list): 
    with open(cut_list, 'rb') as f:
        out = pickle.load(f)
    return out

# --------------------------
# CSV IO (LOG)
# --------------------------
def save_log(df, save_dir, rec_name): 
    csv_path = os.path.join(save_dir, "{}.csv".format(rec_name))
    # write the combined DataFrame to the same CSV file
    df.to_csv(csv_path, index=False)

def load_log(csv_path): 
    df = pd.read_csv(csv_path)
    return df

# --------------------------
# MFCC IO
# --------------------------

def save_list_of_mfcc(mfcc_list, save_dir, rec_name, df): 
    mfcc_names = []
    for i in range(len(mfcc_list)): 
        # loop through the list of mfccs and save them to the directory
        mfcc = mfcc_list[i]
        mfcc_name = "{}_{:08d}.mfcc".format(rec_name, i)
        mfcc_path = os.path.join(save_dir, mfcc_name)
        with open(mfcc_path, 'wb') as f:
            pickle.dump(mfcc, f)
        mfcc_names.append(mfcc_name)
    df.insert(1, "mfcc_name", mfcc_names, allow_duplicates=True)
    return df

def save_mfcc(mfcc_list, save_dir, rec_name): 
    with open(os.path.join(save_dir, "{}.mfcc".format(rec_name)), 'wb') as f:
        pickle.dump(mfcc_list, f)

def load_mfcc(mfcc_path): 
    with open(mfcc_path, 'rb') as f:
        out = pickle.load(f)
    return out

# --------------------------
# Train List IO
# --------------------------
def append_to_csv(csv_path, new_df):
    # read in the existing CSV file as a DataFrame
    existing_df = pd.read_csv(csv_path)
    
    # concatenate the existing DataFrame with the new DataFrame
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # write the combined DataFrame to the same CSV file
    combined_df.to_csv(csv_path, index=False)
    return 


### 重点！有一个trainlist这样才能知道要加载哪些文件
### This trainlist will be like this
### rec_name (in original rec) cutname (of mfcc) start_time end_time token (phone/seq=p1@p2@.../no)