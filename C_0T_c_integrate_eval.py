import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from C_0X_defs import *

def read_result_at(res_save_dir, epoch): 
    all_handler = DictResHandler(whole_res_dir=res_save_dir, 
                                 file_prefix=f"all-{epoch}")

    all_handler.read()

    return all_handler.res

def plot_attention_trajectory_together(all_phi_type, all_attn, all_sepframes1, all_sepframes2, save_path, title="Comparison of Foreign-Attention Trajectory"): 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    legend_namess = [['S-to-P', 'P-to-S', 'P-to-V', 'V-to-P'], ['#-to-P', 'P-to-#', 'P-to-V', 'V-to-P']]
    colors = ['b', 'g', 'red', 'orange']
    n_steps = 100

    for (selector, ax, legend_names) in zip(["ST", "T"], [ax1, ax2], legend_namess):
        selected_tuples = [(sf1, sf2, attn) for pt, sf1, sf2, attn in zip(all_phi_type,  
                                                          all_sepframes1, 
                                                          all_sepframes2, 
                                                          all_attn) if pt == selector]
        selected_sf1s, selected_sf2s, selected_attns = zip(*selected_tuples)
        if selector == "ST":
            s_to_t_traj = []
            t_to_s_traj = []
            t_to_a_traj = []
            a_to_t_traj = []
            for i in range(len(selected_attns)): 
                this_attn = selected_attns[i]
                this_sep_frame1 = selected_sf1s[i]
                this_sep_frame2 = selected_sf2s[i]

                blocks = extract_attention_blocks_ST(this_attn, this_sep_frame1, this_sep_frame2)

                s_to_t_interp = interpolate_traj(blocks['s_to_t'], n_steps)
                t_to_s_interp = interpolate_traj(blocks['t_to_s'], n_steps)
                t_to_a_interp = interpolate_traj(blocks['t_to_a'], n_steps)
                a_to_t_interp = interpolate_traj(blocks['a_to_t'], n_steps)
                if np.any(np.isnan(s_to_t_interp)) or np.any(np.isnan(t_to_s_interp)) or np.any(np.isnan(t_to_a_interp)) or np.any(np.isnan(a_to_t_interp)):
                    print("ST NaN detected!")
                    continue
                s_to_t_traj.append(s_to_t_interp)
                t_to_s_traj.append(t_to_s_interp)
                t_to_a_traj.append(t_to_a_interp)
                a_to_t_traj.append(a_to_t_interp)

            # Convert list of arrays into 2D NumPy arrays for easier manipulation
            group1_array = np.array(s_to_t_traj)
            group2_array = np.array(t_to_s_traj)
            group3_array = np.array(t_to_a_traj)
            group4_array = np.array(a_to_t_traj)

            # Calculate the mean trajectory for each group
            means = np.array([np.mean(group1_array, axis=0), 
                            np.mean(group2_array, axis=0), 
                            np.mean(group3_array, axis=0), 
                            np.mean(group4_array, axis=0)])

            # Calculate the SEM for each step in both groups
            sems = np.array([sem(group1_array, axis=0),
                            sem(group2_array, axis=0),
                            sem(group3_array, axis=0),
                            sem(group4_array, axis=0)])

            # Calculate the 95% CI for both groups
            ci_95s = 1.96 * sems

            # Upper and lower bounds of the 95% CI for both groups
            upper_bounds = means + ci_95s
            lower_bounds = means - ci_95s

            for mean, upper, lower, label, c in zip(means, upper_bounds, lower_bounds, legend_names, colors):
                ax.plot(mean, label=label, color=c)
                ax.fill_between(range(n_steps), lower, upper, alpha=0.2, color=c)

        elif selector == "T": 
            s_to_t_traj = []
            t_to_s_traj = []
            t_to_a_traj = []
            a_to_t_traj = []
            for i in range(len(selected_attns)): 
                this_attn = selected_attns[i]
                this_sep_frame1 = selected_sf1s[i]
                this_sep_frame2 = selected_sf2s[i]

                blocks = extract_attention_blocks_ST(this_attn, this_sep_frame1, this_sep_frame2)

                s_to_t_interp = interpolate_traj(blocks['s_to_t'], n_steps)
                t_to_s_interp = interpolate_traj(blocks['t_to_s'], n_steps)
                t_to_a_interp = interpolate_traj(blocks['t_to_a'], n_steps)
                a_to_t_interp = interpolate_traj(blocks['a_to_t'], n_steps)
                if np.any(np.isnan(s_to_t_interp)) or np.any(np.isnan(t_to_s_interp)) or np.any(np.isnan(t_to_a_interp)) or np.any(np.isnan(a_to_t_interp)):
                    print("T NaN detected!")
                    continue
                s_to_t_traj.append(s_to_t_interp)
                t_to_s_traj.append(t_to_s_interp)
                t_to_a_traj.append(t_to_a_interp)
                a_to_t_traj.append(a_to_t_interp)

            # Convert list of arrays into 2D NumPy arrays for easier manipulation
            group1_array = np.array(s_to_t_traj)
            group2_array = np.array(t_to_s_traj)
            group3_array = np.array(t_to_a_traj)
            group4_array = np.array(a_to_t_traj)

            # Calculate the mean trajectory for each group
            means = np.array([np.mean(group1_array, axis=0), 
                            np.mean(group2_array, axis=0), 
                            np.mean(group3_array, axis=0), 
                            np.mean(group4_array, axis=0)])

            # Calculate the SEM for each step in both groups
            sems = np.array([sem(group1_array, axis=0),
                            sem(group2_array, axis=0),
                            sem(group3_array, axis=0),
                            sem(group4_array, axis=0)])

            # Calculate the 95% CI for both groups
            ci_95s = 1.96 * sems

            # Upper and lower bounds of the 95% CI for both groups
            upper_bounds = means + ci_95s
            lower_bounds = means - ci_95s

            for mean, upper, lower, label, c in zip(means, upper_bounds, lower_bounds, legend_names, colors):
                ax.plot(mean, label=label, color=c)
                ax.fill_between(range(n_steps), lower, upper, alpha=0.2, color=c)
        else: 
            t_to_a_traj = []
            a_to_t_traj = []
            for i in range(len(selected_attns)): 
                this_attn = selected_attns[i]
                this_sep_frame2 = selected_sf2s[i]

                blocks = extract_attention_blocks_T(this_attn, this_sep_frame2)

                t_to_a_interp = interpolate_traj(blocks['t_to_a'], n_steps)
                a_to_t_interp = interpolate_traj(blocks['a_to_t'], n_steps)
                t_to_a_traj.append(t_to_a_interp)
                a_to_t_traj.append(a_to_t_interp)

            # Convert list of arrays into 2D NumPy arrays for easier manipulation
            group3_array = np.array(t_to_a_traj)
            group4_array = np.array(a_to_t_traj)

            # Calculate the mean trajectory for each group
            means = np.array([np.mean(group3_array, axis=0), 
                            np.mean(group4_array, axis=0)])

            # Calculate the SEM for each step in both groups
            sems = np.array([sem(group3_array, axis=0),
                            sem(group4_array, axis=0)])

            # Calculate the 95% CI for both groups
            ci_95s = 1.96 * sems

            # Upper and lower bounds of the 95% CI for both groups
            upper_bounds = means + ci_95s
            lower_bounds = means - ci_95s

            for mean, upper, lower, label, c in zip(means, upper_bounds, lower_bounds, legend_names[2:], colors[2:]):
                ax.plot(mean, label=label, color=c)
                ax.fill_between(range(n_steps), lower, upper, alpha=0.2, color=c)
        ax.set_xlabel('Normalized Time')
        ax.set_ylabel('Summed Foreign-Attention')
        ax.set_title(f'{selector}')
        ax.set_ylim([0, 1])
        ax.legend(loc = "upper left")
        ax.grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--timestamp', '-ts', type=str, default="0000000000", help="Timestamp for project, better be generated by bash")
    parser.add_argument('--gpu', '-gpu', type=int, default=0, help="Choose the GPU to work on")
    parser.add_argument('--model','-m',type=str, default = "ae",help="Model type: ae or vqvae")
    parser.add_argument('--condition','-cd',type=str, default="b", help='Condition: b (balanced), u (unbalanced), nt (no-T)')
    parser.add_argument('--startepoch','-se',type=int, default=0, help='Condition: b (balanced), u (unbalanced), nt (no-T)')
    parser.add_argument('--endepoch','-ee',type=int, default=100, help='Condition: b (balanced), u (unbalanced), nt (no-T)')
    args = parser.parse_args()

    # set device number
    torch.cuda.set_device(args.gpu)

    ts = args.timestamp # this timestamp does not contain run number
    model_type = args.model
    model_condition = args.condition
    train_name = "C_0T"
    res_save_dir = os.path.join(model_save_, f"eval-{train_name}-{ts}")

    sil_dict = {}
    model_condition_dir = os.path.join(res_save_dir, model_type, model_condition)
    print(model_condition_dir)
    assert PU.path_exist(model_condition_dir)
    this_save_dir = os.path.join(model_condition_dir, "integrated_results")
    mk(this_save_dir)

    every_attns = []
    every_attns_pp = []
    every_sepframes1 = []
    every_sepframes2 = []
    every_phi_types = []

    # this is added because some runs failed to do any reconstruction. 
    learned_runs = [1, 2, 3, 4, 5]
    string_learned_runs = [str(num) for num in learned_runs]
    strseq_learned_runs = "".join(string_learned_runs)
    startepoch = args.startepoch
    endepoch = args.endepoch
    print(f"Start epoch: {startepoch}, End epoch: {endepoch}")

    for epoch in range(startepoch, endepoch): 
        cat_attns = []
        cat_attns_pp = []
        cat_sepframes1 = []
        cat_sepframes2 = []
        cat_phi_types = []
        print(f"Processing {model_type} at {epoch}...")

        for run_number in learned_runs:
            this_model_condition_dir = os.path.join(model_condition_dir, f"{run_number}")
            if not PU.path_exist(this_model_condition_dir): 
                print(f"Run {run_number} does not exist.")
                continue
            allres = read_result_at(this_model_condition_dir, epoch)
            cat_phi_types += allres["phi-type"]
            cat_attns += allres["attn"]
            # cat_attns_pp += allres["attn-pp"]
            cat_sepframes1 += allres["sep-frame1"]
            cat_sepframes2 += allres["sep-frame2"]

        # plot_attention_trajectory_together(cat_phi_types, cat_attns, cat_sepframes1, cat_sepframes2, os.path.join(this_save_dir, f"attntraj-at-{epoch}.png"))
        every_attns += cat_attns
        # every_attns_pp += cat_attns_pp
        every_sepframes1 += cat_sepframes1
        every_sepframes2 += cat_sepframes2
        every_phi_types += cat_phi_types
    plot_attention_trajectory_together(every_phi_types, every_attns, every_sepframes1, every_sepframes2, os.path.join(res_save_dir, f"attntraj-at-all-{model_type}-{model_condition}-{strseq_learned_runs}-{startepoch}-{endepoch}.png"), 
                                       f"Comparison of Foreign-Attention Trajectory for Epochs {startepoch} to {endepoch}")
    # plot_attention_trajectory_together(every_phi_types, every_attns_pp, every_sepframes1, every_sepframes2, os.path.join(res_save_dir, f"attnpptraj-at-all-{model_type}-{model_condition}-{strseq_learned_runs}.png"))

    print("Done.")