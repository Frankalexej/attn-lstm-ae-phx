# attn-lstm-ae-phx
Attention-LSTM Autoencoder Learns Phonotactics

## Experiment Codes
The C_0T prefixed files are the experiment runners, evaluators and plotters for the results presented in the poster. 

## How to set up enviroment
1. Create directory and its subdirectory "script"
2. Put all files in repo into script
3. use conda env create -f environment.yml to set up environment, if you have anaconda
4. run python paths.py to set up src directories
5. preproc_ prefixed files are used in preprocessing
6. misc_ prefixed files are other utils
7. debug_ prefixed files are for development
8. model_ prefixed files are for modelling
9. test_ prefixed files are for testing stage (not debugging)

## Data
This version uses the LibriSpeech dataset, accompanied by third-party TextGrid annotation files generated with MFA. The TGs contains word-level and phoneme-level annotation. 

## Data Preprocessing
Two tasks: cut the audio according to annotation and organize information into an integrated file. 
- preproc_guide_extract.py: use it if you only want to exract the metadata but not touch the recordings

1. preproc_seg.py: run this file and get the continuous recordings cut into phones or words  
2. preproc_guide_integrate.py: run and integrate the guide files into one large guideline  
3. preproc_guide_mod.py: run and make additional changes to the guide. You can self-define any change because this is post-hoc. Currently it includes "destress" (disregarding the stress markings), "addpath" (add extra path combined from rec and idx. This will take around twice the size in storage but will save calculation time when loading dataset), "markspeaker" (separately mark the speaker of the recording, this can help with later separation of training, validation and testing sets)  
4. preproc_guide_sepTVT.py: separate training, validation as well as test dataset. This will make sure any speaker (not only segments) is only in one of the sets.

## How to Run
Please kindly refer to the C_0T prefixed files. They are ordered with small letters. For example, C_0T_a_xxx is the first, C_0T_b_xxx is the second.   
When there is a .sh file, you may call the .sh file, as it is written for multiprocessing and one-bag parameter control. 
