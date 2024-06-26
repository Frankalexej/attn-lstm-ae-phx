import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from C_0X_defs import *
from C_0Y_evaldefs import *


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    # ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
    ax.imshow(specgram, origin="lower", aspect="auto", interpolation="nearest")

def get_endframes(seppos, attn_size): 
    return [0, seppos], [seppos, attn_size]

def create_phoneme_block_matrix(starts, ends, total):
    # Initialize an empty list to store phoneme block matrices
    phoneme_blocks = []
    # Iterate through the phoneme frames to create each block
    for start_frame, end_frame in list(zip(starts, ends))[:-1]:
        num_frames = end_frame - start_frame
        phoneme_block = np.ones((num_frames, num_frames))
        phoneme_blocks.append(phoneme_block)
    num_frames = total - starts[-1]
    phoneme_block = np.ones((num_frames, num_frames))
    phoneme_blocks.append(phoneme_block)
    block_diag_matrix = block_diag(*phoneme_blocks)
    return block_diag_matrix

def post2pre_filter(start, sep, end): 
    return np.block([[np.zeros((sep-start, end))], [np.ones((end-sep, sep-start)), np.zeros((end-sep, end-sep))]])

def biway_filter(start, sep, end): 
    return np.block([[np.zeros((sep-start, sep-start)), np.ones((sep-start, end-sep))], [np.ones((end-sep, sep-start)), np.zeros((end-sep, end-sep))]])

def get_in_phone_attn(attn, starts, ends, total): 
    block_diag_matrix = create_phoneme_block_matrix(starts, ends, total)
    filtered_attn = block_diag_matrix * attn
    in_phoneme_attn = filtered_attn.sum(-1)
    return in_phoneme_attn

def interpolate_traj(current, n_steps=100): 
    current_steps = np.linspace(0, 1, num=len(current))
    target_steps = np.linspace(0, 1, num=n_steps)
    interp_func = interp1d(current_steps, current, kind='linear')
    return interp_func(target_steps)

def cutHid(hid, cutstart, cutend, start_offset=0, end_offset=1): 
    selstart = max(cutstart, math.floor(cutstart + (cutend - cutstart) * start_offset))
    selend = min(cutend, math.ceil(cutstart + (cutend - cutstart) * end_offset))
    # hid is (L, H)
    return hid[selstart:selend, :]

def separate_and_sample_data(data_array, tag_array, sample_size):
    # Ensure data_array and tag_array are numpy arrays
    data_array = np.array(data_array)
    tag_array = np.array(tag_array)
    tags = np.unique(tag_array)
    data_list = []
    tag_list = []
    for tag in tags: 
        filtered_data, filtered_tag = filter_data_by_tags(data_array, tag_array, [tag])
        indices = np.random.choice(len(filtered_data), size=sample_size, replace=(sample_size > len(filtered_data)))
        selected_data = filtered_data[indices]
        selected_tag = filtered_tag[indices]
        data_list.append(selected_data)
        tag_list.append(selected_tag)
    return data_list, tag_list

# we have very limited data, so we don't need to select, just plot all
def get_toplot(hiddens, sepframes1, sepframes2, phi_types, stop_names, offsets=(0, 1), contrast_in="asp", merge=True, hidden_dim=8): 
    # collect the start and end frames for each phoneme
    cutstarts = []
    cutends = []
    for sepframe1, sepframe2, phi_type in zip(sepframes1, sepframes2, phi_types):
        # if phi_type == 'ST':
        #     cutstarts.append(sepframe1)
        # else:
        #     cutstarts.append(0)
        # we deleted the above code because now we have SIL. 
        cutstarts.append(sepframe1)
        cutends.append(sepframe2)

    if contrast_in == "asp": 
        tags_list = phi_types
    elif contrast_in == "stop":
        tags_list = stop_names
    else:
        raise ValueError("Contrast_in must be one of 'asp' or 'stop'")
    
    hid_sel = np.empty((0, hidden_dim))
    tag_sel = []
    for (item, start, end, tag) in zip(hiddens, cutstarts, cutends, tags_list): 
        hid = cutHid(item, start, end, offsets[0], offsets[1])
        if merge:
            hid = np.mean(hid, axis=0, keepdims=True)
            hid_sel = np.concatenate((hid_sel, hid), axis=0)
            tag_sel += [tag]
        else: 
            hidlen = hid.shape[0]
            hid_sel = np.concatenate((hid_sel, hid), axis=0)
            tag_sel += [tag] * hidlen
    return hid_sel, np.array(tag_sel)

def plot_silhouette(silarray_1, silarray_2, save_path): 
    # Convert list of arrays into 2D NumPy arrays for easier manipulation
    # group1_array = np.array(silarray_1)
    # group2_array = np.array(silarray_2)

    group1_array = silarray_1
    group2_array = silarray_2

    n_steps = group1_array.shape[1]
    assert n_steps == group2_array.shape[1]

    # Calculate the mean trajectory for each group
    mean_trajectory_group1 = np.mean(group1_array, axis=0)
    mean_trajectory_group2 = np.mean(group2_array, axis=0)

    # Calculate the SEM for each step in both groups
    sem_group1 = sem(group1_array, axis=0)
    sem_group2 = sem(group2_array, axis=0)

    # Calculate the 95% CI for both groups
    ci_95_group1 = 1.96 * sem_group1
    ci_95_group2 = 1.96 * sem_group2

    # Upper and lower bounds of the 95% CI for both groups
    upper_bound_group1 = mean_trajectory_group1 + ci_95_group1
    lower_bound_group1 = mean_trajectory_group1 - ci_95_group1
    upper_bound_group2 = mean_trajectory_group2 + ci_95_group2
    lower_bound_group2 = mean_trajectory_group2 - ci_95_group2

    # Plotting
    plt.figure(figsize=(12, 8))
    # Mean trajectory for Group 1
    plt.plot(mean_trajectory_group1, label='Aspiration', color='blue')
    # 95% CI area for Group 1
    plt.fill_between(range(n_steps), lower_bound_group1, upper_bound_group1, color='blue', alpha=0.2)
    # Mean trajectory for Group 2
    plt.plot(mean_trajectory_group2, label='Place', color='red')
    # 95% CI area for Group 2
    plt.fill_between(range(n_steps), lower_bound_group2, upper_bound_group2, color='red', alpha=0.2)

    plt.xlabel('Epoch')
    plt.ylabel('Silhouette Score (40%~60%)')
    plt.title('Silhouette Score Across Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--timestamp', '-ts', type=str, default="0000000000", help="Timestamp for project, better be generated by bash")
    parser.add_argument('--gpu', '-gpu', type=int, default=0, help="Choose the GPU to work on")
    parser.add_argument('--model','-m',type=str, default = "ae",help="Model type: ae or vqvae")
    parser.add_argument('--condition','-cd',type=str, default="b", help='Condition: b (balanced), u (unbalanced), nt (no-T)')
    parser.add_argument('--zlevel','-zl',type=str, default="hidrep", help='hidrep / attnout')
    args = parser.parse_args()

    # set device number
    torch.cuda.set_device(args.gpu)

    ts = args.timestamp # this timestamp does not contain run number
    train_name = "C_0T"
    res_save_dir = os.path.join(model_save_, f"eval-{train_name}-{ts}")

    model_type = args.model
    model_condition = args.condition
    model_condition_dir = os.path.join(res_save_dir, model_type, model_condition)
    print(model_condition_dir)
    assert PU.path_exist(model_condition_dir)
    this_save_dir = os.path.join(model_condition_dir, "integrated_results")
    mk(this_save_dir)

    asp_sil_lists = []   # silhouette score between aspirated and deaspirated plosives
    stop_sil_lists = []  # silhouette score between p, t, and k
    if model_type == "recon4-phi": 
        hidden_dim = 4
    elif model_type == "recon8-phi": 
        hidden_dim = 8
    elif model_type == "recon16-phi": 
        hidden_dim = 16
    elif model_type == "recon32-phi": 
        hidden_dim = 32
    else: 
        raise ValueError("Model type must be one of 'recon3-phi', 'recon8-phi', 'recon32-phi', 'recon100-phi'")
    learned_runs = [1, 2, 3, 4, 5]
    string_learned_runs = [str(num) for num in learned_runs]
    strseq_learned_runs = "".join(string_learned_runs)

    for run_number in learned_runs:
        asp_list = []
        stop_list = []
        print(f"Processing {model_type} in run {run_number}...")

        for epoch in range(0, 100):     
            this_model_condition_dir = os.path.join(model_condition_dir, f"{run_number}")
            hidrep_handler = DictResHandler(whole_res_dir=this_model_condition_dir, 
                                 file_prefix=f"all-{epoch}")
            hidrep_handler.read()
            hidrep = hidrep_handler.res
            if args.zlevel == "hidrep": 
                all_zq = hidrep["ze"]
            elif args.zlevel == "attnout": 
                all_zq = hidrep["zq"]
            else: 
                raise ValueError("zlevel must be one of 'hidrep' or 'attnout'")
            
            all_sepframes1 = hidrep["sep-frame1"]
            all_sepframes2 = hidrep["sep-frame2"]
            all_phi_type = hidrep["phi-type"]
            all_stop_names = hidrep["sn"]

            # aspiration contrast
            cluster_groups = ["T", "ST"]
            hidr_cs, tags_cs = get_toplot(hiddens=all_zq, 
                                            sepframes1=all_sepframes1,
                                            sepframes2=all_sepframes2,
                                            phi_types=all_phi_type,
                                            stop_names=all_stop_names,
                                            offsets=(0, 1), 
                                            contrast_in="asp", 
                                            merge=True, 
                                            hidden_dim=hidden_dim)
            color_translate = {item: idx for idx, item in enumerate(cluster_groups)}
            hidr_cs, tags_cs = postproc_standardize(hidr_cs, tags_cs, outlier_ratio=0.5)
            asp_this_epoch = []
            for i in range(6):
                hidrs, tagss = separate_and_sample_data(data_array=hidr_cs, tag_array=tags_cs, sample_size=15) # should be 2 only
                abx_err = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                asp_this_epoch.append(abx_err)
            asp_list.append(asp_this_epoch)

            # Silhouette Score
            cluster_groups = ["P", "T", "K"]
            hidr_cs, tags_cs = get_toplot(hiddens=all_zq, 
                                            sepframes1=all_sepframes1,
                                            sepframes2=all_sepframes2,
                                            phi_types=all_phi_type,
                                            stop_names=all_stop_names,
                                            offsets=(0, 1), 
                                            contrast_in="stop", 
                                            merge=True, 
                                            hidden_dim=hidden_dim)
            color_translate = {item: idx for idx, item in enumerate(cluster_groups)}
            hidr_cs, tags_cs = postproc_standardize(hidr_cs, tags_cs, outlier_ratio=0.5)
            stop_this_epoch = []
            for i in range(2): 
                hidrs, tagss = separate_and_sample_data(data_array=hidr_cs, tag_array=tags_cs, sample_size=15)
                abx_err01 = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                abx_err02 = sym_abx_error(hidrs[0], hidrs[2], distance=euclidean_distance)
                abx_err12 = sym_abx_error(hidrs[1], hidrs[2], distance=euclidean_distance)
                stop_this_epoch.append(abx_err01)
                stop_this_epoch.append(abx_err02)
                stop_this_epoch.append(abx_err12)
            stop_list.append(stop_this_epoch)
        asp_sil_lists.append(asp_list)
        stop_sil_lists.append(stop_list)

    stop_arr = np.array(stop_sil_lists)
    asp_arr = np.array(asp_sil_lists)

    stop_arr_reshaped = stop_arr.transpose(0, 2, 1).reshape(-1, stop_arr.shape[1])
    asp_arr_reshaped = asp_arr.transpose(0, 2, 1).reshape(-1, asp_arr.shape[1])

    plot_silhouette(asp_arr_reshaped, stop_arr_reshaped, os.path.join(res_save_dir, f"abx-VS-{model_type}-{model_condition}-{strseq_learned_runs}@{args.zlevel}.png"))
    print("Done.")