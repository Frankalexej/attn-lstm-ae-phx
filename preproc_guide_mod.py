"""
Other modifications on generated guide
"""
import pandas as pd
import os

from paths import *
from misc_tools import PathUtils as PU
from misc_tools import AudioCut
import argparse

def remove_stress(segment):
    if segment and segment[-1].isdigit():
        return segment[:-1]
    return segment

def stress_value(segment): 
    if segment and segment[-1].isdigit(): 
        return segment[-1]  # this is also str, not int
    return "SNA" # meaning stress not applicable

def modify_destress(src_path, target_path): 
    assert (PU.path_exist(src_path))

    total_df = pd.read_csv(src_path)
    # post-hoc changes
    total_df['segment_nostress'] = total_df['segment'].apply(remove_stress)
    total_df['stress_type'] = total_df['segment'].apply(stress_value)

    total_df.to_csv(target_path, index=False)
    return

def modify_markspeaker(src_path, target_path): 
    # only applicable to matched_phone_guides. Not applicable to unmatched guides. 
    assert (PU.path_exist(src_path))

    total_df = pd.read_csv(src_path)
    total_df['speaker'] = total_df.apply(AudioCut.record2speaker, axis=1)

    total_df.to_csv(target_path, index=False)
    return


def modify_addpath(src_path, target_path): 
    # only applicable to matched_phone_guides. Not applicable to unmatched guides. 
    assert (PU.path_exist(src_path))

    total_df = pd.read_csv(src_path)
    total_df['phone_path'] = total_df.apply(AudioCut.record2filepath, axis=1)
    total_df['word_path'] = total_df.apply(AudioCut.wordrecord2filepath, axis=1)

    total_df.to_csv(target_path, index=False)
    return



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--task', '-t', type=str, default="destress", 
                        choices=['destress', 'addpath', 'markspeaker'], 
                        help="Choose the task to perform")
    parser.add_argument('--guide_path', '-gp', type=str, default="..//src/eng/", help="Path that holds the guide files")
    parser.add_argument('--in_file_name', '-ifn', type=str, default="matched_phone_guide.csv", help="Name of input filename")
    parser.add_argument('--out_file_name', '-ofn', type=str, default="matched_phone_guide.csv", help="Name of output filename")
    args = parser.parse_args()


    if args.task == "destress":
        modify_destress(os.path.join(args.guide_path, args.in_file_name), os.path.join(args.guide_path, args.out_file_name))
    elif args.task == "addpath":
        modify_addpath(os.path.join(args.guide_path, args.in_file_name), os.path.join(args.guide_path, args.out_file_name))
    elif args.task == "markspeaker": 
        modify_markspeaker(os.path.join(args.guide_path, args.in_file_name), os.path.join(args.guide_path, args.out_file_name))