from praatio import textgrid
import pandas as pd
import os

from paths import *
from misc_tools import PathUtils as PU
from tqdm import tqdm

def extract_from_tg(read_path, save_dir, tg_name, level="words", save_small=True): 
    tg = textgrid.openTextgrid(read_path, False)
    entries = tg.getTier(level).entries # Get all intervals

    segment = []    # note down what sound it is
    file = []       # filename of [sentence], not sound
    id = []         # address within sentence file
    startTime = []  # start time of segment
    endTime = []    # end time of segment

    for idx, segment_interval in enumerate(entries): 
        segment.append(segment_interval.label)
        file.append(tg_name)
        id.append(idx)
        startTime.append(segment_interval.start)
        endTime.append(segment_interval.end)
    

    data = {
    'segment': segment,
    'file': file,
    'id': id, 
    'startTime': startTime,
    'endTime': endTime
    }

    if save_small: 
        # Create a Pandas DataFrame
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(save_dir, tg_name + ".csv"), index=False)
    return data


def match_word_phone(words_data, phones_data):
    # Convert words and phones data to DataFrame if they are not already
    df_words = pd.DataFrame(words_data) if not isinstance(words_data, pd.DataFrame) else words_data
    df_phones = pd.DataFrame(phones_data) if not isinstance(phones_data, pd.DataFrame) else phones_data

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

    return df_phones




def extract_from_tgs(src_path, word_target_path, phone_target_path): 
    """
    structure of src_path example: 
    train-clean-100-tg/19/198/19-198-0000.TextGrid
    src_path is the hyperpath of all textgrids
    make sure target_path is existent
    """
    assert (PU.path_exist(word_target_path) and PU.path_exist(phone_target_path) and PU.path_exist(src_path))

    for speaker_ in tqdm(sorted(os.listdir(src_path), key=str.casefold)): 
        # train-clean-100-tg/[19]/198/19-198-0000.TextGrid
        src_speaker_ = os.path.join(src_path, speaker_)
        if not os.path.isdir(src_speaker_): 
            continue
        # prepare speaker-level target paths
        word_tgt_speaker_ = os.path.join(word_target_path, speaker_)
        phone_tgt_speaker_ = os.path.join(phone_target_path, speaker_)
        PU.mk(word_tgt_speaker_)
        PU.mk(phone_tgt_speaker_)
        for rec_ in sorted(os.listdir(src_speaker_), key=str.casefold): 
            src_speaker_rec_ = os.path.join(src_speaker_, rec_)
            # prepare rec-level target paths
            word_tgt_speaker_rec_ = os.path.join(word_tgt_speaker_, rec_)
            phone_tgt_speaker_rec_ = os.path.join(phone_tgt_speaker_, rec_)
            PU.mk(word_tgt_speaker_rec_)
            PU.mk(phone_tgt_speaker_rec_)
            for sentence in sorted(os.listdir(src_speaker_rec_), key=str.casefold): 
                # here we loop through each textgrid file
                word_data = extract_from_tg(
                    read_path=os.path.join(src_speaker_rec_, sentence), 
                    save_dir=word_tgt_speaker_rec_, 
                    tg_name=os.path.splitext(sentence)[0], 
                    level="words",
                )

                phone_data = extract_from_tg(
                    read_path=os.path.join(src_speaker_rec_, sentence), 
                    save_dir=phone_tgt_speaker_rec_, 
                    tg_name=os.path.splitext(sentence)[0], 
                    level="phones",
                )

                # match word and phone
                phone_data = match_word_phone(word_data, phone_data)
                phone_data.to_csv(os.path.join(phone_tgt_speaker_rec_, os.path.splitext(sentence)[0] + ".csv"), index=False)
    return



if __name__ == "__main__": 
    extract_from_tgs(train_tg_, train_cut_word_guide_, train_cut_phone_guide_)