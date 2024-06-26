import os
import pandas as pd
from paths import *
from misc_tools import PathUtils as PU
from misc_multiprocessing import *
import argparse

def match_sentence(word_, phone_, matched_, name, save_small=True): 
    word_file = os.path.join(word_, name + ".csv")
    phone_file = os.path.join(phone_, name + ".csv")

    df_words = pd.read_csv(word_file)
    df_phones = pd.read_csv(phone_file)

    # Add new columns to the phones DataFrame
    df_phones['word_id'] = None
    df_phones['word'] = None
    df_phones['in_id'] = None

    # Iterate over each phone and find matching word
    for idx, phone in df_phones.iterrows():
        matching_words = df_words[(df_words['startTime'] <= phone['startTime']) & (df_words['endTime'] >= phone['endTime'])]
        
        if not matching_words.empty:
            word = matching_words.iloc[0]  # Get the first matching word
            df_phones.at[idx, 'word_id'] = word['id']
            df_phones.at[idx, 'word'] = word['segment']
            # Counting the internal index (in_id) of the phone within the word
            df_phones.at[idx, 'in_id'] = (df_phones[(df_phones['word_id'] == word['id']) & (df_phones.index <= idx)]).shape[0]
    if save_small: 
        df_phones.to_csv(os.path.join(matched_, name + ".csv"), index=False)
    return df_phones


def match_dataset(work_list, dir_word, dir_phone, dir_matched): 
    """
    work_list: list of speakers to work on
    dir_word: directory of word guide csv files
    dir_phone: directory of phone guide csv files
    dir_matched: directory to save matched phone guide csv files
    """
    assert (PU.path_exist(dir_word) \
            and PU.path_exist(dir_phone) \
                and PU.path_exist(dir_matched))  # check dir existence
    

    for speaker_ in work_list: 
        print(speaker_)
        # train-clean-100-audio/[19]/198/19-198-0000.csv
        word_speaker_, phone_speaker_ = os.path.join(dir_word, speaker_), os.path.join(dir_phone, speaker_)
        matched_speaker_ = os.path.join(dir_matched, speaker_)
        # if not (PU.path_isdir(word_speaker_) or PU.path_isdir(phone_speaker_)): 
        #     continue
        PU.mk(matched_speaker_)

        for rec_ in sorted(os.listdir(word_speaker_), key=str.casefold): 
            # train-clean-100-audio/19/[198]/19-198-0000.csv
            word_speaker_rec_, phone_speaker_rec_ = os.path.join(word_speaker_, rec_), os.path.join(phone_speaker_, rec_)
            matched_speaker_rec_ = os.path.join(matched_speaker_, rec_)
            PU.mk(matched_speaker_rec_)

            for sentence in sorted(os.listdir(word_speaker_rec_), key=str.casefold): 
                if not sentence.endswith(".csv"): 
                    continue
                # here we loop through each textgrid file
                data = match_sentence(
                    word_=word_speaker_rec_,
                    phone_=phone_speaker_rec_,
                    matched_=matched_speaker_rec_,
                    name=os.path.splitext(sentence)[0], 
                )

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--num_processes', '-np', type=int, default=64, help="Number of processes")
    args = parser.parse_args()
    run_mp(match_dataset, os.listdir(train_cut_word_guide_), args.num_processes, *(train_cut_word_guide_, 
                                                                                   train_cut_phone_guide_, 
                                                                                   train_cut_matched_phone_guide_))