"""
R is completely the same as E, but we delete PP, and will mainly use the phenomenon dataset to train the model. 

T is replicating R. But with 2 layers only. 
"""
######################### Libs #########################
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_sequence
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
# import summary
from model_model import AEPPV1, AEPPV2, AEPPV4, AEPPV8
from model_dataset import DS_Tools
# this is used for CTC pred
from model_dataset import WordDatasetPhoneseq as TrainDataset
from model_dataset import TargetVowelDatasetPhoneseq as TestDataset
from model_dataset import Normalizer, DeNormalizer, TokenMap, WordDictionary
from model_dataset import MelSpecTransformDB as TheTransform
from paths import *
from misc_my_utils import *
from misc_recorder import *
from model_loss import *
from model_padding import generate_mask_from_lengths_mat

BATCH_SIZE = 512
INPUT_DIM = 64
OUTPUT_DIM = 64 
INTER_DIM_0 = 32
INTER_DIM_1 = 16
INTER_DIM_2 = 8
ENC_SIZE_LIST = [INPUT_DIM, INTER_DIM_0, INTER_DIM_1, INTER_DIM_2]
DEC_SIZE_LIST = [OUTPUT_DIM, INTER_DIM_0, INTER_DIM_1, INTER_DIM_2]
DROPOUT = 0.5
NUM_LAYERS = 2
EMBEDDING_DIM = 128
REC_SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 64
LOADER_WORKER = 32


def random_sample_by_speaker(tg, stg, valid_proportion=0.2): 
    # sample speakers, tg and stg use the same, because they must have the same speaker choice 
    speakerlist = stg["speaker"].unique()
    valid_size = int(len(speakerlist) * valid_proportion)
    sampled_indices = np.random.choice(len(speakerlist), size=valid_size, replace=False)
    valid_speakers = speakerlist[sampled_indices]

    t_stg = stg[~stg["speaker"].isin(valid_speakers)]
    v_stg = stg[stg["speaker"].isin(valid_speakers)]

    t_tg = tg[~tg["speaker"].isin(valid_speakers)]
    v_tg = tg[tg["speaker"].isin(valid_speakers)]

    t_s_tg = t_tg.sample(n=len(t_stg))
    v_s_tg = v_tg.sample(n=len(v_stg))

    return (t_stg, v_stg), (t_tg, v_tg), (t_s_tg, v_s_tg)


def generate_separation(T_path, ST_path, target_path): 
    t_guide = pd.read_csv(T_path)
    st_guide = pd.read_csv(ST_path)
    (training_st, valid_st), (training_t, valid_t), (training_sampled_t, valid_sampled_t) = random_sample_by_speaker(t_guide, st_guide)
    # note that st set is always the same, because it will not undergo any sampling, it is the minor one. 
    training_st.to_csv(os.path.join(target_path, "ST-train.csv"), index=False)
    valid_st.to_csv(os.path.join(target_path, "ST-valid.csv"), index=False)
    training_t.to_csv(os.path.join(target_path, "T-train.csv"), index=False)
    valid_t.to_csv(os.path.join(target_path, "T-valid.csv"), index=False)
    training_sampled_t.to_csv(os.path.join(target_path, "T-train-sampled.csv"), index=False)
    valid_sampled_t.to_csv(os.path.join(target_path, "T-valid-sampled.csv"), index=False)
    return 

def load_data_general(dataset, rec_dir, target_path, load="train", select=0.3, sampled=True, batch_size=1):
    # for general, path is easy, let's just load it
    integrated = pd.read_csv(target_path)
    # integrated = integrated.sample(frac=1).reset_index(drop=True)

    mytrans = TheTransform(sample_rate=REC_SAMPLE_RATE, 
                        n_fft=N_FFT, n_mels=N_MELS, 
                        normalizer=Normalizer.norm_mvn, 
                        denormalizer=DeNormalizer.norm_mvn)
    
    # Load TokenMap to map the phoneme to the index
    with open(os.path.join(src_, "no-stress-seg.dict"), "rb") as file:
        # Load the object from the file
        mylist = pickle.load(file)
        mylist = ["BLANK"] + mylist
        mylist = mylist + ["SIL"]

    # Now you can use the loaded object
    mymap = TokenMap(mylist)

    ds = dataset(rec_dir, 
                        integrated,  
                        mapper=mymap, 
                        transform=mytrans, 
                        ground_truth_path=os.path.join(src_, f"{load}-phoneseq.gt"))
    
    use_len = int(select * len(ds))
    remain_len = len(ds) - use_len
    use_ds, remain_ds = random_split(ds, [use_len, remain_len])

    use_shuffle = True if load == "train" else False
    loader = DataLoader(use_ds, batch_size=batch_size, shuffle=use_shuffle, num_workers=LOADER_WORKER, collate_fn=dataset.collate_fn)
    return loader

def load_data_phenomenon(dataset, rec_dir, target_path, load="train", select="both", sampled=True, batch_size=1):
    if sampled: 
        sample_suffix = "-sampled"
    else:
        sample_suffix = ""

    if select == "both":
        t_set = pd.read_csv(os.path.join(target_path, f"T-{load}{sample_suffix}.csv"))
        st_set = pd.read_csv(os.path.join(target_path, f"ST-{load}.csv"))
        integrated = pd.concat([t_set, st_set], ignore_index=True, sort=False)
    elif select == "T":
        integrated = pd.read_csv(os.path.join(target_path, f"T-{load}{sample_suffix}.csv"))
    elif select == "ST":
        integrated = pd.read_csv(os.path.join(target_path, f"ST-{load}.csv"))
    else: 
        raise ValueError("select must be either both, T or ST")

    integrated = integrated.sample(frac=1).reset_index(drop=True)

    mytrans = TheTransform(sample_rate=REC_SAMPLE_RATE, 
                        n_fft=N_FFT, n_mels=N_MELS, 
                        normalizer=Normalizer.norm_mvn, 
                        denormalizer=DeNormalizer.norm_mvn)
    
    # Load TokenMap to map the phoneme to the index
    with open(os.path.join(src_, "no-stress-seg.dict"), "rb") as file:
        # Load the object from the file
        mylist = pickle.load(file)
        mylist = ["BLANK"] + mylist
        mylist = mylist + ["SIL"]

    # Now you can use the loaded object
    mymap = TokenMap(mylist)

    ds = dataset(rec_dir, 
                        integrated, 
                        mapper=mymap,
                        transform=mytrans)

    use_shuffle = True if load == "train" else False
    loader = DataLoader(ds, batch_size=batch_size, shuffle=use_shuffle, num_workers=LOADER_WORKER, collate_fn=dataset.collate_fn)
    return loader


def draw_learning_curve_and_accuracy(losses, recons, embeddings, commitments, start=0, end=100, save_name=""):
    train_losses, valid_losses, onlyST_valid_losses = losses
    train_recon_losses, valid_recon_losses, onlyST_valid_recon_losses = recons
    train_embedding_losses, valid_embedding_losses, onlyST_valid_embedding_losses = embeddings
    train_commitment_losses, valid_commitment_losses, onlyST_valid_commitment_losses = commitments

    start, end = 0, 100
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 4))
    # Plot Loss on the left subplot
    ax1.plot(train_losses.get()[start:end], label='Train')
    ax1.plot(valid_losses.get()[start:end], label='Valid')
    ax1.plot(onlyST_valid_losses.get()[start:end], label='OnlyST Valid')
    ax1.set_title("Learning Curve Loss")

    # Plot Recon Loss on the right subplot
    ax2.plot(train_recon_losses.get()[start:end], label='Train')
    ax2.plot(valid_recon_losses.get()[start:end], label='Valid')
    ax2.plot(onlyST_valid_recon_losses.get()[start:end], label='OnlyST Valid')
    ax2.set_title("Learning Curve Recon Loss")

    # Plot Embedding Loss on the left subplot
    ax3.plot(train_embedding_losses.get()[start:end], label='Train')
    ax3.plot(valid_embedding_losses.get()[start:end], label='Valid')
    ax3.plot(onlyST_valid_embedding_losses.get()[start:end], label='OnlyST Valid')
    ax3.set_title("Learning Curve Embedding Loss")

    # Plot Commitment Loss on the right subplot
    ax4.plot(train_commitment_losses.get()[start:end], label='Train')
    ax4.plot(valid_commitment_losses.get()[start:end], label='Valid')
    ax4.plot(onlyST_valid_commitment_losses.get()[start:end], label='OnlyST Valid')
    ax4.set_title("Learning Curve Commitment Loss")

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    plt.tight_layout()
    plt.savefig(save_name)
    return 

def initialize_model(model):
    # init LSTM-attn AE
    for name, param in model.named_parameters():
        if 'weight_ih' in name:  # Weights of the input-hidden for LSTM layers
            nn.init.orthogonal_(param.data)
        elif 'weight_hh' in name:  # Weights of the hidden-hidden (recurrent) for LSTM layers
            nn.init.orthogonal_(param.data)
        elif 'weight' in name:  # Weights of linear layers
            layer_type = name.split('.')[0]
            if isinstance(getattr(model, layer_type), nn.Linear):  # Check if the layer is linear
                nn.init.orthogonal_(param.data)


def run_once(hyper_dir, model_type="ae", condition="b"): 
    model_save_dir = os.path.join(hyper_dir, model_type, condition)
    mk(model_save_dir)

    # Loss Recording
    train_losses = ListRecorder(os.path.join(model_save_dir, "train.loss"))
    train_recon_losses = ListRecorder(os.path.join(model_save_dir, "train.recon.loss"))
    train_embedding_losses = ListRecorder(os.path.join(model_save_dir, "train.embedding.loss"))
    train_commitment_losses = ListRecorder(os.path.join(model_save_dir, "train.commitment.loss"))

    valid_losses = ListRecorder(os.path.join(model_save_dir, "valid.loss"))
    valid_recon_losses = ListRecorder(os.path.join(model_save_dir, "valid.recon.loss"))
    valid_embedding_losses = ListRecorder(os.path.join(model_save_dir, "valid.embedding.loss"))
    valid_commitment_losses = ListRecorder(os.path.join(model_save_dir, "valid.commitment.loss"))

    # In C we take onlyST to record the phenomenon-target dataset
    onlyST_valid_losses = ListRecorder(os.path.join(model_save_dir, "valid_onlyST.loss"))
    onlyST_valid_recon_losses = ListRecorder(os.path.join(model_save_dir, "valid_onlyST.recon.loss"))
    onlyST_valid_embedding_losses = ListRecorder(os.path.join(model_save_dir, "valid_onlyST.embedding.loss"))
    onlyST_valid_commitment_losses = ListRecorder(os.path.join(model_save_dir, "valid_onlyST.commitment.loss"))

    text_hist = HistRecorder(os.path.join(model_save_dir, "trainhist.txt"))

    # Recording Directory
    phone_rec_dir = train_cut_phone_
    word_rec_dir = train_cut_word_
    train_guide_path = os.path.join(src_, "guide_train.csv")
    valid_guide_path = os.path.join(src_, "guide_validation.csv")

    # Load TokenMap to map the phoneme to the index
    with open(os.path.join(src_, "no-stress-seg.dict"), "rb") as file:
        # Load the object from the file
        mylist = pickle.load(file)
        mylist = ["BLANK"] + mylist
        mylist = mylist + ["SIL"]   # this is to fit STV vs #TV

    # Now you can use the loaded object
    mymap = TokenMap(mylist)
    class_dim = mymap.token_num()
    ctc_size_list = {'hid': INTER_DIM_2, 'class': class_dim}

    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == "mtl":
        batch_size = 512
        masked_loss = MaskedLoss(loss_fn=nn.MSELoss(reduction="none"))
        ctc_loss = nn.CTCLoss(blank=mymap.encode("BLANK"))
        model_loss = AlphaCombineLoss(masked_loss, ctc_loss, alpha=0.2)

        model = AEPPV1(enc_size_list=ENC_SIZE_LIST, 
                   dec_size_list=DEC_SIZE_LIST, 
                   ctc_decoder_size_list=ctc_size_list,
                   num_layers=NUM_LAYERS, dropout=DROPOUT)
        
        # Load Data
        guide_path = os.path.join(hyper_dir, "guides")
        train_loader = load_data_general(TrainDataset, 
                                        word_rec_dir, train_guide_path, load="train", select=0.15, sampled=False, batch_size=batch_size)
        valid_loader = load_data_general(TrainDataset, 
                                        word_rec_dir, valid_guide_path, load="valid", select=0.15, sampled=False, batch_size=batch_size)
        onlyST_valid_loader = load_data_phenomenon(TestDataset, 
                                                phone_rec_dir, guide_path, load="valid", select="both", sampled=True, batch_size=batch_size)
    elif model_type == "pp": 
        batch_size = 512
        masked_loss = MaskedLoss(loss_fn=nn.MSELoss(reduction="none"))
        ctc_loss = nn.CTCLoss(blank=mymap.encode("BLANK"))
        model_loss = PseudoAlphaCombineLoss_Pred(masked_loss, ctc_loss, alpha=0.2)
        model = AEPPV2(enc_size_list=ENC_SIZE_LIST, 
                   dec_size_list=DEC_SIZE_LIST, 
                   ctc_decoder_size_list=ctc_size_list,
                   num_layers=NUM_LAYERS, dropout=DROPOUT)
        # Load Data
        guide_path = os.path.join(hyper_dir, "guides")
        train_loader = load_data_general(TrainDataset, 
                                        word_rec_dir, train_guide_path, load="train", select=0.3, sampled=False, batch_size=batch_size)
        valid_loader = load_data_general(TrainDataset, 
                                        word_rec_dir, valid_guide_path, load="valid", select=0.3, sampled=False, batch_size=batch_size)
        onlyST_valid_loader = load_data_phenomenon(TestDataset, 
                                                phone_rec_dir, guide_path, load="valid", select="both", sampled=True, batch_size=batch_size)
        
    elif model_type == "mtl-phi": 
        batch_size = 32
        # NOTE: mtl-phi is just training on phenomenon dataset and test on that as well. 
        masked_loss = MaskedLoss(loss_fn=nn.MSELoss(reduction="none"))
        ctc_loss = nn.CTCLoss(blank=mymap.encode("BLANK"))
        model_loss = AlphaCombineLoss(masked_loss, ctc_loss, alpha=0.2)

        model = AEPPV1(enc_size_list=ENC_SIZE_LIST, 
                   dec_size_list=DEC_SIZE_LIST, 
                   ctc_decoder_size_list=ctc_size_list,
                   num_layers=NUM_LAYERS, dropout=DROPOUT)
        
        # Load Data
        guide_path = os.path.join(hyper_dir, "guides")
        train_loader = load_data_phenomenon(TestDataset, 
                                        phone_rec_dir, guide_path, load="train", select="both", sampled=False, batch_size=batch_size)
        valid_loader = load_data_phenomenon(TestDataset, 
                                        phone_rec_dir, guide_path, load="valid", select="both", sampled=True, batch_size=batch_size)
        onlyST_valid_loader = load_data_phenomenon(TestDataset, 
                                                phone_rec_dir, guide_path, load="valid", select="ST", sampled=True, batch_size=batch_size)
    elif model_type == "recon-phi": 
        batch_size = 32
        # NOTE: mtl-phi is just training on phenomenon dataset and test on that as well. 
        masked_loss = MaskedLoss(loss_fn=nn.MSELoss(reduction="none"))
        ctc_loss = nn.CTCLoss(blank=mymap.encode("BLANK"))
        model_loss = PseudoAlphaCombineLoss_Recon(masked_loss, ctc_loss, alpha=0.2)

        model = AEPPV8(enc_size_list=ENC_SIZE_LIST, 
                   dec_size_list=DEC_SIZE_LIST, 
                   ctc_decoder_size_list=ctc_size_list,
                   num_layers=NUM_LAYERS, dropout=DROPOUT)
        
        # Load Data
        guide_path = os.path.join(hyper_dir, "guides")
        train_loader = load_data_phenomenon(TestDataset, 
                                        phone_rec_dir, guide_path, load="train", select="both", sampled=False, batch_size=batch_size)
        valid_loader = load_data_phenomenon(TestDataset, 
                                        phone_rec_dir, guide_path, load="valid", select="both", sampled=True, batch_size=batch_size)
    elif model_type == "recon4-phi": 
        batch_size = 32
        # NOTE: mtl-phi is just training on phenomenon dataset and test on that as well. 
        masked_loss = MaskedLoss(loss_fn=nn.MSELoss(reduction="none"))
        ctc_loss = nn.CTCLoss(blank=mymap.encode("BLANK"))
        model_loss = PseudoAlphaCombineLoss_Recon(masked_loss, ctc_loss, alpha=0.2)
        enc_list = [INPUT_DIM, INTER_DIM_0, INTER_DIM_1, 4]
        dec_list = [OUTPUT_DIM, INTER_DIM_0, INTER_DIM_1, 4]
        model = AEPPV8(enc_size_list=enc_list, 
                   dec_size_list=dec_list, 
                   ctc_decoder_size_list=ctc_size_list,
                   num_layers=NUM_LAYERS, dropout=DROPOUT)
        # Load Data
        guide_path = os.path.join(hyper_dir, "guides")
        train_loader = load_data_phenomenon(TestDataset, 
                                        phone_rec_dir, guide_path, load="train", select="both", sampled=False, batch_size=batch_size)
        valid_loader = load_data_phenomenon(TestDataset, 
                                        phone_rec_dir, guide_path, load="valid", select="both", sampled=True, batch_size=batch_size)
    elif model_type == "recon8-phi": 
        batch_size = 32
        # NOTE: mtl-phi is just training on phenomenon dataset and test on that as well. 
        masked_loss = MaskedLoss(loss_fn=nn.MSELoss(reduction="none"))
        ctc_loss = nn.CTCLoss(blank=mymap.encode("BLANK"))
        model_loss = PseudoAlphaCombineLoss_Recon(masked_loss, ctc_loss, alpha=0.2)
        enc_list = [INPUT_DIM, INTER_DIM_0, INTER_DIM_1, 8]
        dec_list = [OUTPUT_DIM, INTER_DIM_0, INTER_DIM_1, 8]
        model = AEPPV8(enc_size_list=enc_list, 
                   dec_size_list=dec_list, 
                   ctc_decoder_size_list=ctc_size_list,
                   num_layers=NUM_LAYERS, dropout=DROPOUT)
        # Load Data
        guide_path = os.path.join(hyper_dir, "guides")
        train_loader = load_data_phenomenon(TestDataset, 
                                        phone_rec_dir, guide_path, load="train", select="both", sampled=False, batch_size=batch_size)
        valid_loader = load_data_phenomenon(TestDataset, 
                                        phone_rec_dir, guide_path, load="valid", select="both", sampled=True, batch_size=batch_size)
    elif model_type == "recon16-phi": 
        batch_size = 32
        # NOTE: mtl-phi is just training on phenomenon dataset and test on that as well. 
        masked_loss = MaskedLoss(loss_fn=nn.MSELoss(reduction="none"))
        ctc_loss = nn.CTCLoss(blank=mymap.encode("BLANK"))
        model_loss = PseudoAlphaCombineLoss_Recon(masked_loss, ctc_loss, alpha=0.2)
        enc_list = [INPUT_DIM, INTER_DIM_0, INTER_DIM_1, 16]
        dec_list = [OUTPUT_DIM, INTER_DIM_0, INTER_DIM_1, 16]
        model = AEPPV8(enc_size_list=enc_list, 
                   dec_size_list=dec_list, 
                   ctc_decoder_size_list=ctc_size_list,
                   num_layers=NUM_LAYERS, dropout=DROPOUT)
        # Load Data
        guide_path = os.path.join(hyper_dir, "guides")
        train_loader = load_data_phenomenon(TestDataset, 
                                        phone_rec_dir, guide_path, load="train", select="both", sampled=False, batch_size=batch_size)
        valid_loader = load_data_phenomenon(TestDataset, 
                                        phone_rec_dir, guide_path, load="valid", select="both", sampled=True, batch_size=batch_size)
    elif model_type == "recon32-phi": 
        batch_size = 32
        # NOTE: mtl-phi is just training on phenomenon dataset and test on that as well. 
        masked_loss = MaskedLoss(loss_fn=nn.MSELoss(reduction="none"))
        ctc_loss = nn.CTCLoss(blank=mymap.encode("BLANK"))
        model_loss = PseudoAlphaCombineLoss_Recon(masked_loss, ctc_loss, alpha=0.2)
        enc_list = [INPUT_DIM, INTER_DIM_0, INTER_DIM_1, 32]
        dec_list = [OUTPUT_DIM, INTER_DIM_0, INTER_DIM_1, 32]
        model = AEPPV8(enc_size_list=enc_list, 
                   dec_size_list=dec_list, 
                   ctc_decoder_size_list=ctc_size_list,
                   num_layers=NUM_LAYERS, dropout=DROPOUT)
        # Load Data
        guide_path = os.path.join(hyper_dir, "guides")
        train_loader = load_data_phenomenon(TestDataset, 
                                        phone_rec_dir, guide_path, load="train", select="both", sampled=False, batch_size=batch_size)
        valid_loader = load_data_phenomenon(TestDataset, 
                                        phone_rec_dir, guide_path, load="valid", select="both", sampled=True, batch_size=batch_size)
    else: 
        raise Exception("Model type not supported! ")

    model.to(device)
    initialize_model(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model_str = str(model)
    model_txt_path = os.path.join(model_save_dir, "model.txt")
    with open(model_txt_path, "w") as f:
        f.write(model_str)
        f.write("\n")

    num_epochs = 100

    for epoch in range(num_epochs):
        text_hist.print("Epoch {}".format(epoch))
        model.train()
        train_loss = 0.
        train_cumulative_l_reconstruct = 0.
        train_cumulative_l_embedding = 0.
        train_cumulative_l_commitment = 0.
        train_num = len(train_loader.dataset)    # train_loader
        for idx, ((x, y_preds), (x_lens, y_preds_lens), pt, sn) in enumerate(train_loader):
            current_batch_size = x.shape[0]
            # y_lens should be the same as x_lens
            optimizer.zero_grad()
            x_mask = generate_mask_from_lengths_mat(x_lens, device=device)
            y_recon = x
            x = x.to(device)
            y_recon = y_recon.to(device)
            y_preds = y_preds.to(device)
            y_preds = y_preds.long()

            (x_hat_recon, y_hat_preds), (attn_w_recon, attn_w_preds), (ze, zq) = model(x, x_lens, x_mask)
            y_hat_preds = y_hat_preds.permute(1, 0, 2)

            l_alpha, (l_reconstruct, l_prediction) = model_loss.get_loss(x_hat_recon, y_recon, 
                                                                         y_hat_preds, y_preds, 
                                                                         x_lens, y_preds_lens, 
                                                                         x_mask)
            
            loss, l_reconstruct, l_embedding, l_commitment = l_alpha, l_reconstruct, l_prediction, l_prediction

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * current_batch_size
            train_cumulative_l_reconstruct += l_reconstruct.item() * current_batch_size
            train_cumulative_l_embedding += l_embedding.item() * current_batch_size
            train_cumulative_l_commitment += l_commitment.item() * current_batch_size

            if idx % 100 == 0:
                text_hist.print(f"""Training step {idx} loss {loss: .3f} \t recon {l_reconstruct: .3f} \t embed {l_embedding: .3f} \t commit {l_commitment: .3f}""")

        train_losses.append(train_loss / train_num)
        train_recon_losses.append(train_cumulative_l_reconstruct / train_num)
        train_embedding_losses.append(train_cumulative_l_embedding / train_num)
        train_commitment_losses.append(train_cumulative_l_commitment / train_num)

        # text_hist.print(f"""※※※Training loss {train_loss / train_num: .3f} \t recon {train_cumulative_l_reconstruct / train_num: .3f} \t embed {train_cumulative_l_embedding / train_num: .3f} \t commit {train_cumulative_l_commitment / train_num: .3f}※※※""")

        last_model_name = "{}.pt".format(epoch)
        torch.save(model.state_dict(), os.path.join(model_save_dir, last_model_name))

        # Valid (ST + T)
        model.eval()
        valid_loss = 0.
        valid_cumulative_l_reconstruct = 0.
        valid_cumulative_l_embedding = 0.
        valid_cumulative_l_commitment = 0.
        valid_num = len(valid_loader.dataset)
        for idx, ((x, y_preds), (x_lens, y_preds_lens), pt, sn) in enumerate(valid_loader):
            current_batch_size = x.shape[0]
            x_mask = generate_mask_from_lengths_mat(x_lens, device=device)

            y_recon = x
            x = x.to(device)
            y_recon = y_recon.to(device)
            y_preds = y_preds.to(device)
            y_preds = y_preds.long()

            (x_hat_recon, y_hat_preds), (attn_w_recon, attn_w_preds), (ze, zq) = model(x, x_lens, x_mask)
            y_hat_preds = y_hat_preds.permute(1, 0, 2)

            l_alpha, (l_reconstruct, l_prediction) = model_loss.get_loss(x_hat_recon, y_recon, 
                                                                         y_hat_preds, y_preds, 
                                                                         x_lens, y_preds_lens, 
                                                                         x_mask)
            loss, l_reconstruct, l_embedding, l_commitment = l_alpha, l_reconstruct, l_prediction, l_prediction

            valid_loss += loss.item() * current_batch_size
            valid_cumulative_l_reconstruct += l_reconstruct.item() * current_batch_size
            valid_cumulative_l_embedding += l_embedding.item() * current_batch_size
            valid_cumulative_l_commitment += l_commitment.item() * current_batch_size

        # text_hist.print(f"""※※※Valid loss {valid_loss / valid_num: .3f} \t recon {valid_cumulative_l_reconstruct / valid_num: .3f} \t embed {valid_cumulative_l_embedding / valid_num: .3f} \t commit {valid_cumulative_l_commitment / valid_num: .3f}※※※""")
        valid_losses.append(valid_loss / valid_num)
        valid_recon_losses.append(valid_cumulative_l_reconstruct / valid_num)
        valid_embedding_losses.append(valid_cumulative_l_embedding / valid_num)
        valid_commitment_losses.append(valid_cumulative_l_commitment / valid_num)

        # # Valid (ST)
        # model.eval()
        # onlyST_valid_loss = 0.
        # onlyST_valid_cumulative_l_reconstruct = 0.
        # onlyST_valid_cumulative_l_embedding = 0.
        # onlyST_valid_cumulative_l_commitment = 0.
        # onlyST_valid_num = len(onlyST_valid_loader.dataset)
        # for idx, ((x, y_preds), (x_lens, y_preds_lens), pt, sn) in enumerate(onlyST_valid_loader):
        #     current_batch_size = x.shape[0]
        #     x_mask = generate_mask_from_lengths_mat(x_lens, device=device)

        #     y_recon = x
        #     x = x.to(device)
        #     y_recon = y_recon.to(device)
        #     y_preds = y_preds.to(device)
        #     y_preds = y_preds.long()

        #     (x_hat_recon, y_hat_preds), (attn_w_recon, attn_w_preds), (ze, zq) = model(x, x_lens, x_mask)
        #     y_hat_preds = y_hat_preds.permute(1, 0, 2)

        #     l_alpha, (l_reconstruct, l_prediction) = model_loss.get_loss(x_hat_recon, y_recon, 
        #                                                                  y_hat_preds, y_preds, 
        #                                                                  x_lens, y_preds_lens, 
        #                                                                  x_mask)
        #     loss, l_reconstruct, l_embedding, l_commitment = l_alpha, l_reconstruct, l_prediction, l_prediction

        #     onlyST_valid_loss += loss.item() * current_batch_size
        #     onlyST_valid_cumulative_l_reconstruct += l_reconstruct.item() * current_batch_size
        #     onlyST_valid_cumulative_l_embedding += l_embedding.item() * current_batch_size
        #     onlyST_valid_cumulative_l_commitment += l_commitment.item() * current_batch_size

        # # text_hist.print(f"""※※※OnlyST Valid loss {onlyST_valid_loss / onlyST_valid_num: .3f} \t recon {onlyST_valid_cumulative_l_reconstruct / onlyST_valid_num: .3f} \t embed {onlyST_valid_cumulative_l_embedding / onlyST_valid_num: .3f} \t commit {onlyST_valid_cumulative_l_commitment / onlyST_valid_num: .3f}※※※""")
        # onlyST_valid_losses.append(onlyST_valid_loss / onlyST_valid_num)
        # onlyST_valid_recon_losses.append(onlyST_valid_cumulative_l_reconstruct / onlyST_valid_num)
        # onlyST_valid_embedding_losses.append(onlyST_valid_cumulative_l_embedding / onlyST_valid_num)
        # onlyST_valid_commitment_losses.append(onlyST_valid_cumulative_l_commitment / onlyST_valid_num)

        train_losses.save()
        train_recon_losses.save()
        train_embedding_losses.save()
        train_commitment_losses.save()
        valid_losses.save()
        valid_recon_losses.save()
        valid_embedding_losses.save()
        valid_commitment_losses.save()
        onlyST_valid_losses.save()
        onlyST_valid_recon_losses.save()
        onlyST_valid_embedding_losses.save()
        onlyST_valid_commitment_losses.save()


        if epoch % 20 == 0:
            draw_learning_curve_and_accuracy(losses=(train_losses, valid_losses, onlyST_valid_losses), 
                                            recons=(train_recon_losses, valid_recon_losses, onlyST_valid_recon_losses),
                                            embeddings=(train_embedding_losses, valid_embedding_losses, onlyST_valid_embedding_losses),
                                            commitments=(train_commitment_losses, valid_commitment_losses, onlyST_valid_commitment_losses),
                                            start=0, end=epoch,
                                            save_name=os.path.join(model_save_dir, "vis.png"))

    draw_learning_curve_and_accuracy(losses=(train_losses, valid_losses, onlyST_valid_losses), 
                                    recons=(train_recon_losses, valid_recon_losses, onlyST_valid_recon_losses),
                                    embeddings=(train_embedding_losses, valid_embedding_losses, onlyST_valid_embedding_losses),
                                    commitments=(train_commitment_losses, valid_commitment_losses, onlyST_valid_commitment_losses),
                                    start=0, end=epoch,
                                    save_name=os.path.join(model_save_dir, "vis.png"))



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--dataprepare', '-dp', action="store_true")
    parser.add_argument('--timestamp', '-ts', type=str, default="0000000000", help="Timestamp for project, better be generated by bash")
    parser.add_argument('--gpu', '-gpu', type=int, default=0, help="Choose the GPU to work on")
    parser.add_argument('--model','-m',type=str, default = "ae",help="Model type: ae or vqvae")
    parser.add_argument('--condition','-cd',type=str, default="b", help='Condition: b (balanced), u (unbalanced), nt (no-T)')
    args = parser.parse_args()

    ## Hyper-preparations
    ts = args.timestamp
    train_name = "C_0T"
    model_save_dir = os.path.join(model_save_, f"{train_name}-{ts}")
    mk(model_save_dir) 

    if args.dataprepare: 
        # data prepare
        print(f"{train_name}-{ts}-DataPrepare")
        guide_path = os.path.join(model_save_dir, "guides")
        mk(guide_path)
        generate_separation(os.path.join(src_, "phi-T-guide.csv"), 
                            os.path.join(src_, "phi-ST-guide.csv"), 
                            guide_path)

    else: 
        print(f"{train_name}-{ts}")
        torch.cuda.set_device(args.gpu)
        run_once(model_save_dir, model_type=args.model, condition=args.condition)