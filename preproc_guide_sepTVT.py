"""
Other modifications on generated guide
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np

from paths import *
from misc_tools import PathUtils as PU
import argparse


# def modify_guide(src_path, target_dir, target_prefix): 
#     assert (PU.path_exist(src_path))

#     total_df = pd.read_csv(src_path)

#     speakers = total_df["file"].str.split("-").str[0].unique().tolist()

#     np.random.shuffle(speakers)
#     num_speakers = len(speakers)
#     ratios = [0.8, 0.1, 0.1]
#     train_speakers = speakers[:int(num_speakers*ratios[0])]
#     val_speakers = speakers[int(num_speakers*ratios[0]):int(num_speakers*(ratios[0] + ratios[1]))]
#     test_speakers = speakers[int(num_speakers*(ratios[0] + ratios[1])):]
    
#     # create output tensors for each set of speakers
#     train_df = total_df[total_df["file"].str.split("-").str[0].isin(train_speakers)]
#     val_df = total_df[total_df["file"].str.split("-").str[0].isin(val_speakers)]
#     test_df = total_df[total_df["file"].str.split("-").str[0].isin(test_speakers)]

#     train_df.to_csv(os.path.join(target_dir, target_prefix + "train.csv"), index=False)
#     val_df.to_csv(os.path.join(target_dir, target_prefix + "validation.csv"), index=False)
#     test_df.to_csv(os.path.join(target_dir, target_prefix + "test.csv"), index=False)
#     return

# def split_dataset_by_speaker(file_path, train_size=0.7, val_size=0.15, test_size=0.15):
#     # Read the CSV file
#     df = pd.read_csv(file_path)

#     # Initialize lists to hold the split data
#     train_data = []
#     val_data = []
#     test_data = []

#     # Group by speaker
#     grouped = df.groupby('speaker')

#     # Iterate over each group and split the data
#     for _, group in grouped:
#         # Splitting the group into train and temp sets
#         train, temp = train_test_split(group, train_size=train_size, random_state=42)

#         # Splitting the temp set into validation and test sets
#         val, test = train_test_split(temp, test_size=test_size/(val_size + test_size), random_state=42)

#         # Append to the respective lists
#         train_data.append(train)
#         val_data.append(val)
#         test_data.append(test)

#     # Concatenate all dataframes in the lists
#     train_df = pd.concat(train_data)
#     val_df = pd.concat(val_data)
#     test_df = pd.concat(test_data)

#     # Return the split dataframes
#     return train_df, val_df, test_df

def split_dataset_by_speaker(src_path, target_dir, target_prefix): 
    assert (PU.path_exist(src_path))

    total_df = pd.read_csv(src_path)

    speakers = total_df["speaker"].unique().tolist()

    np.random.shuffle(speakers)
    num_speakers = len(speakers)
    ratios = [0.8, 0.1, 0.1]
    train_speakers = speakers[:int(num_speakers*ratios[0])]
    val_speakers = speakers[int(num_speakers*ratios[0]):int(num_speakers*(ratios[0] + ratios[1]))]
    test_speakers = speakers[int(num_speakers*(ratios[0] + ratios[1])):]
    
    # create output tensors for each set of speakers
    train_df = total_df[total_df["speaker"].isin(train_speakers)]
    val_df = total_df[total_df["speaker"].isin(val_speakers)]
    test_df = total_df[total_df["speaker"].isin(test_speakers)]

    train_df.to_csv(os.path.join(target_dir, target_prefix + "train.csv"), index=False)
    val_df.to_csv(os.path.join(target_dir, target_prefix + "validation.csv"), index=False)
    test_df.to_csv(os.path.join(target_dir, target_prefix + "test.csv"), index=False)
    return


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--guide_path', '-gp', type=str, default="..//src/eng/", help="Path that holds the guide files")
    parser.add_argument('--in_file_name', '-ifn', type=str, default="matched_phone_guide.csv", help="Name of input filename")
    parser.add_argument('--out_file_name', '-ofn', type=str, default="guide_", help="Name of input filename")
    args = parser.parse_args()

    split_dataset_by_speaker(os.path.join(args.guide_path, args.in_file_name), args.guide_path, args.out_file_name)
    # train_df.to_csv(os.path.join(args.guide_path, "training_" + args.in_file_name), index=False)
    # val_df.to_csv(os.path.join(args.guide_path, "validation_" + args.in_file_name), index=False)
    # test_df.to_csv(os.path.join(args.guide_path, "testing_" + args.in_file_name), index=False)