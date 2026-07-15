import math
import pandas as pd


def create_word_frequency_dict():
    word_freq_table = pd.read_csv("./data/wordFreqDict.csv").fillna(1)
    word_freq_table["Frequency"] = word_freq_table["Frequency"].astype(int)
    return word_freq_table


def cost_function():
    return [0, 0]


def cost_function_inverse_log_frequency(message_frequencies, freq_max=800000, beta=1):
    """
    diff beta values for native/nonnative speakers

    message_frequencies should just be a list of frequencies of the words
    since the dictionary is in the model class
    """
    return [math.log(freq_max ** 2 / (beta + freq)) for freq in message_frequencies]


def cost_function_log_frequency(message_frequencies, beta=1):
    """
    diff beta values for native/nonnative speakers

    message_frequencies should just be a list of frequencies of the words
    since the dictionary is in the model class
    """
    return [(beta - math.log(freq)) + 14 for freq in message_frequencies]


def cost_function_inverse_cube_root(message_frequencies, beta=1):
    return [1 / (beta * freq) ** (1/3) for freq in message_frequencies]


def frequency_costs(word_freq_table, words):
    cost_list = []
    for word in words:
        cost_list.extend(list(word_freq_table[word_freq_table["Word"] == word].Frequency))

    return cost_list
