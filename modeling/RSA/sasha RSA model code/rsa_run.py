from RSA.model_definition import RSAFrequencyModel, create_word_frequency_dict
from RSA.main import cost_function_by_frequency, frequency_costs, cost_function_inverse_cube_root

import math
import pandas as pd

alpha = 1
uniform_priors = [1/3, 1/3, 1/3]

model = RSAFrequencyModel(priors=uniform_priors, alpha=alpha, cost_function=cost_function_inverse_cube_root)
# model = RSAFrequencyModel(priors=priors, alpha=alpha, cost_function=cost_function_by_frequency)
word_frequency_table = create_word_frequency_dict()


stimuli = pd.read_csv("./data/final-stimuli.csv")
stimuli = stimuli[["Weak", "Strong", "Antonym"]]
# remove the two practice trials
messages = stimuli.values[2:]


# min value is 431
all_words = pd.read_csv("./data/wordFreqDict.csv").fillna(300)
all_words = all_words[["Word", "Frequency"]]
freq_max = all_words["Frequency"].max()
all_words = all_words.values

beta_values = [.00001, .0001, .001, .005, .01, .05, .1, .5, 1, 2]

# # Plotting Some Curves for Diff Cost Functions
# costs = []
# for word, freq in all_words:
#     # beta_costs = [math.log(freq_max * 2 / (beta + freq)) for beta in beta_values]
#     beta_costs = [1 / (beta * freq) ** (1/3) for beta in beta_values]
#     costs.append([freq, *beta_costs])
#
# # df = pd.DataFrame(costs, columns=["word", "beta", "freq", "cost"])
#
# df = pd.DataFrame(costs, columns=["freq", *beta_values])
#
# df.to_csv("./output/costCalcs-log-cuberoot.csv", index=False)


# Simulating some predictions
preds = []
for beta in beta_values:
    for message in messages[:1]:
        message_freq = frequency_costs(word_frequency_table, list(message))
        freq_ratio = math.log10(message_freq[0] / message_freq[1])
        l1_values = model.get_L1_values(message, message, beta=beta)
        preds.append([beta, *message_freq[:2], float(l1_values[0][0]), float(l1_values[0][1])])

df = pd.DataFrame(preds, columns=["beta", "freq_weak", "freq_strong", "p(weak|weak)", "p(strong|weak)"])
df.to_csv("./data/predictions.csv")


# # Generating Graphs of the L1 Values
# df = pd.DataFrame(preds, columns=["beta", "nativeness", "weakScalar", "strongScalar"])
# print(df.head())
# for obj, msg in zip(stimuli, stimuli):
#     # model.plot_L1(obj, msg)
#     model.compare_L1_values(obj, msg)

