import os
import torchaudio
from praatio import textgrid
import pandas as pd
from paths import *
from misc_tools import PathUtils as PU, AudioCut
from misc_multiprocessing import *
import argparse

# xxx_ means xxx_path (self-defined abbr)
def segment_and_extract_sentence(au_, tg_, ca_, cg_, name, level="words", save_small=True): 
    # in current setting save_small is always true
    audio_file = os.path.join(au_, name + ".flac")
    tg_file = os.path.join(tg_, name + ".TextGrid")

    tg = textgrid.openTextgrid(tg_file, False)
    entries = tg.getTier(level).entries # Get all intervals

    segment = []    # note down what sound it is
    file = []       # filename of [sentence], not sound
    id = []         # address within sentence file
    startTime = []  # start time of segment
    endTime = []    # end time of segment
    nSample = []    # number of samples in recording [not calculated]

    speaker, rec, sentence = AudioCut.solve_name(name)

    ca_name_ = os.path.join(ca_, sentence)
    PU.mk(ca_name_)

    for idx, segment_interval in enumerate(entries): 
        starttime = segment_interval.start
        endtime = segment_interval.end

        # Extract and save audio segment
        waveform, sample_rate = torchaudio.load(audio_file)
        start_sample = AudioCut.time2frame(starttime, sample_rate)
        end_sample = AudioCut.time2frame(endtime, sample_rate)
        cut_audio = waveform[:, start_sample:end_sample]

        ca_file = os.path.join(ca_name_, AudioCut.cut_name_gen(name, idx, 4))
        torchaudio.save(ca_file, cut_audio, sample_rate)

        segment.append(segment_interval.label)
        file.append(name)
        id.append(idx)
        startTime.append(starttime)
        endTime.append(endtime)
        nSample.append(cut_audio.shape[1])
    

    data = {
    'segment': segment,
    'file': file,
    'id': id, 
    'startTime': startTime,
    'endTime': endTime, 
    'nSample': nSample
    }

    if save_small: 
        # Create a Pandas DataFrame
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(cg_, name + ".csv"), index=False)
    return data


def segment_and_extract(work_list, dir_au, dir_tg, dir_ca, dir_cg, level="words"): 
    """
    This function reads textgrid files, and according to the 
    interval boundaries cut the corresponding recordings into small audios. 
    In the mean time note down the metadata needed for training. 
    """
    assert (PU.path_exist(dir_au) \
            and PU.path_exist(dir_tg) \
                and PU.path_exist(dir_ca) \
                and PU.path_exist(dir_cg))  # check dir existence
    
    # total_speakers = len(work_list)
    # sorted(os.listdir(dir_au), key=str.casefold)

    for speaker_ in work_list: 
        print(speaker_)
        # train-clean-100-audio/[19]/198/19-198-0000.flac
        audio_speaker_, tg_speaker_ = os.path.join(dir_au, speaker_), os.path.join(dir_tg, speaker_)
        if not (PU.path_isdir(audio_speaker_) or PU.path_isdir(tg_speaker_)): 
            continue

        ca_speaker_, cg_speaker_ = os.path.join(dir_ca, speaker_), os.path.join(dir_cg, speaker_)
        PU.mk(ca_speaker_)
        PU.mk(cg_speaker_)

        for rec_ in sorted(os.listdir(audio_speaker_), key=str.casefold): 
            audio_speaker_rec_, tg_speaker_rec_ = os.path.join(audio_speaker_, rec_), os.path.join(tg_speaker_, rec_)

            ca_speaker_rec_, cg_speaker_rec_ = os.path.join(ca_speaker_, rec_), os.path.join(cg_speaker_, rec_)
            PU.mk(ca_speaker_rec_)
            PU.mk(cg_speaker_rec_)

            for sentence in sorted(os.listdir(audio_speaker_rec_), key=str.casefold): 
                if not sentence.endswith(".flac"): 
                    continue
                # here we loop through each textgrid file
                data = segment_and_extract_sentence(
                    au_=audio_speaker_rec_, 
                    tg_=tg_speaker_rec_, 
                    ca_=ca_speaker_rec_, 
                    cg_=cg_speaker_rec_, 
                    name=os.path.splitext(sentence)[0], 
                    level=level,
                )

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--level', '-l', type=str, default="words", help="Cut into words or phones")
    parser.add_argument('--num_processes', '-np', type=int, default=64, help="Number of processes")
    args = parser.parse_args()
    # run_mp(segment_and_extract, os.listdir(train_audio_), 32, *(train_audio_, train_tg_, train_cut_audio_, train_cut_guide_))
    if args.level == "words": 
        run_mp(segment_and_extract, os.listdir(train_audio_), args.num_processes, *(train_audio_, train_tg_, train_cut_word_, train_cut_word_guide_, args.level))
    elif args.level == "phones": 
        run_mp(segment_and_extract, os.listdir(train_audio_), args.num_processes, *(train_audio_, train_tg_, train_cut_phone_, train_cut_phone_guide_, args.level))
    # segment_and_extract(os.listdir(train_audio_), train_audio_, train_tg_, train_cut_audio_, train_cut_guide_)
    # segment_and_extract(try_audio_, try_tg_, try_cut_audio_, try_cut_guide_)