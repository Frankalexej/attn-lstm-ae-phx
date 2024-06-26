import pandas as pd
import os

from paths import *
from misc_tools import PathUtils as PU
from tqdm import tqdm
import argparse

def remove_stress(segment):
    if segment and segment[-1].isdigit():
        return segment[:-1]
    return segment

def integrate_guides(src_path, target_filename): 
    """
    structure of src_path example: 
    train-clean-100-tg/19/198/19-198-0000.TextGrid
    src_path is the hyperpath of all textgrids
    make sure target_path is existent
    """
    assert (PU.path_exist(src_path))

    total_df = pd.DataFrame()
    total_speakers = len(os.listdir(src_path))

    for speaker_ in tqdm(sorted(os.listdir(src_path), key=str.casefold)): 
        # train-clean-100-tg/[19]/198/19-198-0000.TextGrid
        src_speaker_ = os.path.join(src_path, speaker_)
        if not os.path.isdir(src_speaker_): 
            continue
        for rec_ in sorted(os.listdir(src_speaker_), key=str.casefold): 
            src_speaker_rec_ = os.path.join(src_speaker_, rec_)
            for sentence in sorted(os.listdir(src_speaker_rec_), key=str.casefold): 
                # here we loop through each csv guide file
                small_guide_df = pd.read_csv(os.path.join(src_speaker_rec_, sentence))
                total_df = pd.concat([total_df, small_guide_df], ignore_index=True)

    total_df.to_csv(target_filename, index=False)
    return



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--level', '-l', type=str, default="words", help="Cut into words or phones")
    args = parser.parse_args()
    
    if args.level == "words": 
        guide_, target_ = train_cut_word_guide_, src_
        integrate_guides(guide_, os.path.join(target_, "word_guide.csv"))
    elif args.level == "phones": 
        guide_, target_ = train_cut_phone_guide_, src_
        integrate_guides(guide_, os.path.join(target_, "phone_guide.csv"))
    elif args.level == "matched_phones": 
        guide_, target_ = train_cut_matched_phone_guide_, src_
        integrate_guides(guide_, os.path.join(target_, "matched_phone_guide.csv"))