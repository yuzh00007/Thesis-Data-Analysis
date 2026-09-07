from model import RSAFrequencyModel, create_word_frequency_dict
from helper import (
    cost_function_log_frequency,
    cost_function_inverse_log_frequency,
    cost_function_inverse_cube_root,
    frequency_costs
)

import pandas as pd
from scipy.stats import pearsonr


def hyperparam_search(betas, alpha_vals, human_results):
    """
    betas could be nonnative or native
    """
    scores = []
    all_preds = {}
    preds_loc = 0
    for alpha_val in alpha_vals:
        model.set_alpha(alpha_val)

        for beta in betas:
            preds = []
            for index, msg in enumerate(messages):
                msg_frq = frequency_costs(word_frequency_table, list(msg))
                l1_values = model.get_L1_values(msg, msg, beta=beta)
                preds.append([index + 1, *msg_frq[:2], float(l1_values[0][0]), float(l1_values[0][1])])

            df = pd.DataFrame(
                preds,
                columns=["itemId", "weak_freq", "strong_freq", "p(weak|weak)", "p(strong|weak)"]
            )

            df = df.merge(human_results, on="itemId")
            all_preds.update({
                preds_loc: df
            })
            preds_loc += 1

            r_val = pearsonr(df["p(weak|weak)"], df["derivationStrength"])
            scores.append([alpha_val, beta, r_val.statistic, r_val.pvalue])

    results = pd.DataFrame(
        scores, columns=["alpha", "beta", "r_val", "p_value"]
    )
    return results, all_preds


def final_beta_runs(beta_value, alpha_value, human_results):
    model.set_alpha(alpha_value)

    preds = []
    for index, msg in enumerate(messages):
        l1_values = model.get_L1_values(msg, msg, beta_value)
        preds.append([index + 1, float(l1_values[0][0])])

    df = pd.DataFrame(
        preds,
        columns=["itemId", "p(weak|weak)"]
    )

    df = df.merge(human_results, on="itemId")
    return df


if __name__ == "__main__":
    word_frequency_table = create_word_frequency_dict()

    critical_only = True
    effect_only = False

    stimuli = pd.read_csv("./data/final-stimuli.csv")
    stimuli = stimuli[stimuli["Type"] != "practice"]
    if critical_only:
        stimuli = stimuli[stimuli["Type"] == "critical"]
    stimuli = stimuli[["Weak", "Strong", "Antonym"]]
    messages = stimuli.values

    # min value is 431, give NAs a value slightly lower (Zipf's law)
    # hardcoded the wordFreqDict so they have rank 20000 and frequency 400
    all_words = pd.read_csv("./data/wordFreqDict.csv").fillna(400)
    all_words = all_words[["Word", "Frequency"]]
    freq_max = all_words["Frequency"].max()

    cost_function = cost_function_inverse_cube_root
    if not effect_only:
        nonnative_grid = [.00001,]    # alpha = 1.3
        native_grid = [.000008, .000006, .000007]    # alpha = 1.5
    else:
        # alpha for both = 1.500000
        nonnative_grid = [.000014]
        native_grid = [.000006]

    alpha_grid = [.7, .9, 1.1, 1.3, 1.5]
    uniform_priors = [1/3, 1/3, 1/3]
    model = RSAFrequencyModel(priors=uniform_priors, alpha=alpha_grid[0], cost_function=cost_function)

    final_results = pd.read_csv("./data/derivationByItemSpeaker.csv", index_col=[0])
    final_results["derivationStrength"] = final_results["derivationStrength"] / 100
    final_results["strongRating"] = final_results["strongRating"] / 100
    if effect_only:
        final_results = final_results[final_results["derivationStrength"] > final_results["strongRating"]]
    if critical_only:
        final_results = final_results[final_results["itemType"] == "critical"]

    nonnative_results = final_results[final_results["speakerType"] == "nonnative"]
    native_results = final_results[final_results["speakerType"] == "native"]

    ####################
    # BETA GRID SEARCH #
    ####################

    # NONNATIVE
    # nonnat_r_results, nonnat_preds = hyperparam_search(nonnative_grid, alpha_grid, nonnative_results)
    # nonnat_r_results.to_csv("./output/nonnat/r_vals.csv", index=False)
    # nonnat_r_results["r_val_abs"] = abs(nonnat_r_results["r_val"])
    # best_alpha_beta_idx = nonnat_r_results["r_val_abs"].idxmax()
    # best_alpha_beta = nonnat_r_results.loc[best_alpha_beta_idx]
    # print(best_alpha_beta)
    # nonnat_preds[best_alpha_beta_idx].to_csv("./output/nonnat/bestBetaPreds.csv")
    #
    """ RESULTS - ALL CRIT
    alpha        1.300000e+00
    beta         5.400000e-07
    r_val        2.766665e-01
    p_value      2.995857e-01
    r_val_abs    2.766665e-01
    """

    """ RESULTS - EFFECT ONLY
    alpha        1.500000
    beta         0.000014
    r_val        0.275122
    p_value      0.386782
    r_val_abs    0.275122
    """

    """ RESULTS - ALL TRIALS
    alpha        1.500000
    beta         0.000010
    r_val        0.634066
    p_value      0.000098
    r_val_abs    0.634066
    """
    #
    # # NATIVE
    # native_r_results, nativ_preds = hyperparam_search(native_grid, alpha_grid, native_results)
    # native_r_results.to_csv("./output/native/r_vals.csv", index=False)
    # native_r_results["r_val_abs"] = abs(native_r_results["r_val"])
    # best_alpha_beta_idx = native_r_results["r_val_abs"].idxmax()
    # best_alpha_beta = native_r_results.loc[best_alpha_beta_idx]
    # print(best_alpha_beta)
    # nativ_preds[best_alpha_beta_idx].to_csv("./output/native/bestBetaPreds.csv")
    #
    """ RESULTS - ALL CRIT
    alpha        1.500000
    beta         0.125000
    r_val       -0.143208
    p_value      0.596733
    r_val_abs    0.143208
    """

    """ RESULTS - EFFECT ONLY
    alpha        1.500000
    beta         0.000006
    r_val        0.184113
    p_value      0.528640
    r_val_abs    0.184113
    """

    """ RESULTS - ALL TRIALS
    alpha        1.500000
    beta         0.000007
    r_val        0.612524
    p_value      0.000194
    r_val_abs    0.612524
    """

    # OPTIMIZED RUNS
    if not effect_only and not critical_only:
        nonnat_alpha = 1.5
        nonnat_beta = .00001

        nat_alpha = 1.5
        nat_beta = 0.000007

        file_ext = "allTrials"
    elif not effect_only:
        nonnat_alpha = 1.3
        nonnat_beta = 5.4e-07

        nat_alpha = 1.5
        nat_beta = 0.125

        file_ext = "allCrit"
    else:
        nonnat_alpha = 1.5
        nonnat_beta = 0.000014

        nat_alpha = 1.5
        nat_beta = 0.000006

        file_ext = "effectOnly"

    final_beta_runs(
        nonnat_beta, nonnat_alpha, nonnative_results
    ).to_csv(f"./output/nonnat/humanVsModel{file_ext}.csv")
    final_beta_runs(
        nat_beta, nat_alpha, native_results
    ).to_csv(f"./output/native/humanVsModel{file_ext}.csv")


    # Simulating some predictions
    # predictions = []
    # for itemId, message in enumerate(messages):
    #     # TODO create a lookup for freq-ratios instead of calculating it every time
    #     message_freq = frequency_costs(word_frequency_table, list(message))
    #     freq_ratio = math.log10(message_freq[0] / message_freq[1])
    #     native_l1_values = model.get_L1_values(message, message, beta=native_beta)
    #     nonnative_l1_values = model.get_L1_values(message, message, beta=nonnative_beta)
    #
    #     predictions.append([itemId, *message_freq[:2], float(native_l1_values[0][0]), float(nonnative_l1_values[0][0])])
    #
    # df = pd.DataFrame(
    #     predictions,
    #     columns=["itemid", "freq_weak", "freq_strong", "derivationNative", "derivationNotNative"]
    # )
    # df.to_csv("./output/predictions.csv", index=False)


