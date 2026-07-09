from RSA.model_definition import *
import csv
import math


def cost_function():
    return [0, 0]


def cost_function_by_frequency(message_frequencies, beta=1):
    """
    diff beta values for native/nonnative speakers

    message_frequencies should just be a list of frequencies of the words
    since the dictionary is in the model class
    """
    return [(beta - math.log(x)) if x else 5 for x in message_frequencies]


def cost_function_inverse_cube_root(message_frequencies, beta=1):
    return [1 / (beta * freq) ** (1/3) for freq in message_frequencies]

    # return [1 / (beta * freq) ** (1/3) for beta in beta_values]


def frequency_costs(word_freq_table, words):
    cost_list = []
    for word in words:
        cost_list.extend(list(word_freq_table[word_freq_table["Word"] == word].Frequency))

    return cost_list


def initial_lamb_distr():
    lambdas = [x for x in np.arange(0, 1.01, 0.01)]
    lambdas_probs = {l: 1 / len(lambdas) for l in lambdas}

    return lambdas_probs


def mix_lambs(distr1, distr2, weight1=0.5):
    assert distr1.keys() == distr2.keys()
    assert weight1 >= 0 and weight1 <= 1

    mixed_distr = {}

    for lamb in distr1:
        mixed_distr[lamb] = weight1 * distr1[lamb] + (1 - weight1) * distr2[lamb]

    return mixed_distr


def compute_expected_lamb(distr):
    exp_lamb_array = []
    return sum([i * distr[i] for i in distr])


def power_lamb_prior(alpha=1, favor="high"):
    """
    alpha = strength of skew
    favor: high or low, whether low or high lambda values should be more probable
    """
    lambs = [x for x in np.arange(0, 1.01, 0.01)]

    if favor == "high":
        w = [l ** alpha for l in lambs]
    elif favor == "low":
        w = [(1 - l) ** alpha for l in lambs]

    w = w / sum(w)
    return dict(zip(lambs, w))


# TODO change to pandas later
def stimuli_loader(f='./data/adaptation_trials.csv'):
    # should return lists of dictionaries for S0 and S1
    stims = csv.reader(open(f))
    next(stims)

    d_s0 = []
    d_s1 = []

    for row in stims:
        block = row[5]
        o = row[0]
        m = row[3]
        objects = [row[0], row[1]]
        messages = [x for x in row[4].strip().split(',') if (x in objects[0] or x in objects[1])]
        trial_type = row[7]
        target_is_pragmatic = row[8]
        type = 'critical_pragm' if target_is_pragmatic == 'TRUE' else 'critical_nonpragm'
        itemid = row[10]

        observation = {
            'type': type,
            'objects': objects,
            'messages': messages,
            'm': m,
            'o': o,
            'itemid': itemid
        }

        # these don't really matter since we don't expect an update for them under either model
        if trial_type in ['unambiguous', 'filler_b']:
            continue
        else:
            if block == 's_0':
                d_s0.append(observation)
            else:
                assert block == 's_1'
                d_s1.append(observation)

    return (d_s0, d_s1)
