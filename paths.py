import os

root_ = "../"   # hyperpath of here
# using english as default, to facilitate code transferbnd_detect_tools
src_ = root_ + "/src/eng/"
train_audio_ = src_ + "train-clean-100-audio/"
train_tg_ = src_ + "train-clean-100-tg/"
train_cut_word_ = src_ + "train-clean-100-cw/"  # cut words, ideally following structure of original audio
train_cut_word_guide_ = src_ + "train-clean-100-cwg/"  # cut guide, ideally following structure of original audio
train_cut_phone_ = src_ + "train-clean-100-cp/"  # cut phones, phones will have a word idx marking to which word it belongs
train_cut_phone_guide_ = src_ + "train-clean-100-cpg/"
train_cut_matched_phone_guide_ = src_ + "train-clean-100-cmpg/" # store the phone guides after matching with words

debug_ = src_ + "debug/"

model_save_ = root_ + "model_save/"



def mk(dir): 
    os.makedirs(dir, exist_ok = True)


if __name__ == '__main__':
    # For all paths defined, run mk()
    for name, value in globals().copy().items():
        if isinstance(value, str) and not name.startswith("__"):
            globals()[name] = mk(value)