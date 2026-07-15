from model import RSAFrequencyModel, create_word_frequency_dict
from helper import (
    cost_function_log_frequency,
    cost_function_inverse_log_frequency,
    cost_function_inverse_cube_root,
    frequency_costs
)

import math
import pandas as pd


def simulate_predictions(beta_values):
    # Simulating some predictions
    preds = []
    for beta in beta_values:
        for msg in messages[:1]:
            msg_frq = frequency_costs(word_frequency_table, list(msg))
            l1_values = model.get_L1_values(msg, msg, beta=beta)
            preds.append([beta, *msg_frq[:2], float(l1_values[0][0]), float(l1_values[0][1])])

    df = pd.DataFrame(preds, columns=["beta", "freq_weak", "freq_strong", "p(weak|weak)", "p(strong|weak)"])
    df.to_csv("./output/oneBetaPredictions.csv")


if __name__ == "__main__":
    word_frequency_table = create_word_frequency_dict()

    stimuli = pd.read_csv("./data/final-stimuli.csv")
    stimuli = stimuli[["Weak", "Strong", "Antonym"]]
    # remove the two practice trials
    messages = stimuli.values[2:]
    print(messages)

    # min value is 431, give NAs a value slightly lower (Zipf's law)
    # hardcoded the wordFreqDict so they have rank 20000 and frequency 400
    all_words = pd.read_csv("./data/wordFreqDict.csv").fillna(400)
    all_words = all_words[["Word", "Frequency"]]
    freq_max = all_words["Frequency"].max()

    # TODO: finalize these here
    cost_function = cost_function_inverse_cube_root
    nonnative_beta = .001
    native_beta = .1

    # TODO hyperparamatrize alpha
    alpha = 1
    uniform_priors = [1/3, 1/3, 1/3]
    model = RSAFrequencyModel(priors=uniform_priors, alpha=alpha, cost_function=cost_function)

    # beta_grid = [4, 1, .1, .01, .001]
    # simulate_predictions(beta_grid)

    # Simulating some predictions
    predictions = []
    for itemId, message in enumerate(messages):
        # TODO create a lookup for freq-ratios instead of calculating it every time
        message_freq = frequency_costs(word_frequency_table, list(message))
        freq_ratio = math.log10(message_freq[0] / message_freq[1])
        native_l1_values = model.get_L1_values(message, message, beta=native_beta)
        nonnative_l1_values = model.get_L1_values(message, message, beta=nonnative_beta)

        predictions.append([itemId, *message_freq[:2], float(native_l1_values[0][0]), float(nonnative_l1_values[0][0])])

    df = pd.DataFrame(
        predictions,
        columns=["itemid", "freq_weak", "freq_strong", "derivationNative", "derivationNotNative"]
    )
    df.to_csv("./output/predictions.csv", index=False)


