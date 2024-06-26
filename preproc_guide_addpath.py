"""
Other modifications on generated guide
"""
import pandas as pd
import os

from paths import *
from misc_tools import PathUtils as PU
from misc_tools import AudioCut
import argparse

def modify_guide(src_path, target_path): 
    assert (PU.path_exist(src_path))

    total_df = pd.read_csv(src_path)
    total_df['combined_path'] = total_df.apply(AudioCut.record2filepath, axis=1)

    total_df.to_csv(target_path, index=False)
    return



if __name__ == "__main__": 
    # modify_guide(os.path.join(src_, "guide.csv"), os.path.join(src_, "guide_mod.csv"))
    modify_guide(os.path.join(ssrc_, "guide_mod.csv"), os.path.join(ssrc_, "guide_mod_pathed.csv"))