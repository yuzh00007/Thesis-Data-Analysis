from RSA.main import frequency_costs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RSAUpdatingModel:
    """
    simple mixture model with Bayesian belief updating to learn the mixture weight from interaction
    """

    def __init__(self, alpha, priors, cost_function):
        self.alpha = alpha
        self.priors = priors
        self.cost_function = cost_function

    """
    function to normalize an array by dividing each row by its sum
    to obtain a probability distribution
    """
    def normalize(self, arr):
        return arr / arr.sum(axis=1)[:, np.newaxis]

    """
    [[m]](o) -- literal meaning of utterances
    dimensions: 2 messages x 2 objects <- I think??? Or flipped?
    """

    def L0_truth_table(self, objects, messages):
        t = np.zeros(shape=[len(messages), len(objects)])
        for i in range(len(messages)):
            for j in range(len(objects)):
                # check whether the message describes object correctly
                if messages[i] in objects[j]:
                    t[i, j] = 1

        # t = np.array([
        #          [1,1],
        #          [0,1]
        #          ])

        return t

    """
    literal listener L0(o|m)
    matrix dimensions: 3 messages x 2 objects
    """

    def L0(self, objects, messages):
        # compute [[m]](o): literal meaning
        literal_meaning = self.L0_truth_table(objects=objects, messages=messages)

        # P(o) -- create a matrix of priors by repeating the priors array for each row
        priors = np.tile(np.array(self.priors), (2, 1))

        # multiply the two of L0 terms together
        unnorm = literal_meaning * priors

        # normalize to obtain a probability distribution
        norm = self.normalize(unnorm)

        return norm

    """
    literal speaker S0(m|o)
    matrix dimensions: 2 objects x 2 messages

    So I think we can take the L0 truth table and transpose it?
    """

    def S0(self, objects, messages):
        unnorm = np.zeros(shape=[len(objects), len(messages)])
        for i in range(len(objects)):
            for j in range(len(messages)):
                # check whether the message describes object correctly
                if messages[j] in objects[i]:
                    unnorm[i, j] = 1

        norm = self.normalize(unnorm)

        return norm

    """
    pragmatic speaker S1(m|o)
    matrix dimensions: 2 objects x 2 messages
    """

    def S1(self, objects, messages):
        # compute the costs for each expression in messages
        costs = self.cost_function()

        # utility without costs
        raw_utility = np.log(self.L0(objects=objects, messages=messages).T)

        # Cost(m) -- reshape to obtain correct dimensions for matrix multiplication
        costs = np.repeat(self.cost_function(), 2).reshape((2, 2)).T

        # compute utility by subtracting costs from informativity
        utility = np.subtract(raw_utility, costs)

        # compute S1(m|o) by taking the exponent of utility times the temperature parameter alpha
        unnorm = np.exp(self.alpha * utility)

        # normalize to obtain a probability distribution
        norm = self.normalize(unnorm)

        return norm

    """
    mixture speaker Smixture(m|o)
    matrix dimensions: 2 objects x 2 messages
    """

    def S_mix(self, objects, messages, lamb):
        # S0(m|o)
        S0 = self.S0(objects=objects, messages=messages)

        # S1(m|o)
        S1 = self.S1(objects=objects, messages=messages)

        S_mixture_unnorm = lamb * S1 + (1 - lamb) * S0

        S_mixture = self.normalize(S_mixture_unnorm)

        return S_mixture

    """
    pragmatic belief-driven listener L2(o|m)
    matrix dimensions: 2 messages x 2 objects
    """

    def L2_mixture(self, objects, messages, lamb):
        S_mixture = self.S_mix(objects=objects, messages=messages, lamb=lamb)

        # P(o) -- create a matrix of priors by repeating the priors array for each row
        priors = np.tile(np.array(self.priors), (2, 1))

        # multiply S_mixture(m|o) (transformed, since we need a distribution over objects, not over messages)
        # and P(o) together
        unnorm = S_mixture.T * priors

        # normalize to obtain a probability distribution
        norm = self.normalize(unnorm)

        return norm

    """
    Learning lambda from data based on S_mixture
    """

    def compute_lamb_posteriors_based_on_S_mixed(self, lamb_priors, d):
        # d = [{"objects":["tr_re","tr_bl"],"messages":["tr","bl"],"o":"tr_bl","u":"tr"}]
        # lambdas_probs = {0.0: 0.09, 0.1: 0.09, ..., 1.0: 0.09}
        lamb_posteriors = {}

        # for every lambda, compute the posterior
        for lamb in lamb_priors:
            lamb_posterior = lamb_priors[lamb]
            # for that, compute a mixture model for the evidence with that lambda, then update
            for trial in d:
                S_mixture = self.S_mix(objects=trial["objects"], messages=trial["messages"], lamb=lamb)
                # now find the right value; s_mix is a matrix 2 objects x 2 messages

                o_index = trial["objects"].index(trial["o"])
                m_index = trial["messages"].index(trial["m"])

                S_m_given_o = S_mixture[o_index, m_index]

                lamb_posterior *= S_m_given_o

            lamb_posteriors[lamb] = lamb_posterior

        # now renormalize lamb_posteriors to sum up to 1
        sum_posteriors = sum(lamb_posteriors.values())
        lamb_posteriors_norm = {x: lamb_posteriors[x] / sum_posteriors for x in lamb_posteriors}

        return lamb_posteriors_norm

    """
    single updating step
    return posterior distribution over lambdas and the marginal L2 interpetaiton
    """

    def one_lamb_posterior_update_based_on_L2(self, lamb_priors, trial, pragm_target):
        # trial: {"objects":["tr_re","tr_bl"],"messages":["tr","bl"],"o":"tr_bl","u":"tr"}
        # lambdas_probs = {0.0: 0.09, 0.1: 0.09, ..., 1.0: 0.09}

        l2_lambda_free = 0
        lamb_posteriors = {}

        for lamb in lamb_priors:
            lamb_posterior = lamb_priors[lamb]
            L_mixture = self.L2_mixture(objects=trial["objects"], messages=trial["messages"], lamb=lamb)

            o_index = trial["objects"].index(trial["o"])
            m_index = trial["messages"].index(trial["m"])

            L_o_given_m_target = L_mixture[m_index, o_index]
            L_o_given_m_pragm_target = L_mixture[m_index, trial["objects"].index(pragm_target)]

            # for our probability calculation, we want the pragmatic target
            l2_lambda_free += lamb_priors[lamb] * L_o_given_m_pragm_target

            # for our posterior calculation, we want the target the speaker meant, i.e. actual observation
            lamb_posterior *= L_o_given_m_target

            lamb_posteriors[lamb] = lamb_posterior

        # now renormalize lamb_posteriors to sum up to 1
        sum_posteriors = sum(lamb_posteriors.values())
        lamb_posteriors_norm = {x: lamb_posteriors[x] / sum_posteriors for x in lamb_posteriors}

        return l2_lambda_free, lamb_posteriors_norm

    def compute_lamb_posteriors_based_on_L2(self, lamb_priors, d):
        # d = [{"objects":["tr_re","tr_bl"],"messages":["tr","bl"],"o":"tr_bl","u":"tr"}]
        # lambdas_probs = {0.0: 0.09, 0.1: 0.09, ..., 1.0: 0.09}
        lamb_posteriors = {}

        # for every lambda, compute the posterior
        for lamb in lamb_priors:
            lamb_posterior = lamb_priors[lamb]
            # for that, compute a mixture model for the evidence with that lambda, then update
            for trial in d:
                L_mixture = self.L2_mixture(objects=trial["objects"], messages=trial["messages"], lamb=lamb)
                # now find the right value; s_mix is a matrix 2 objects x 2 messages

                o_index = trial["objects"].index(trial["o"])
                m_index = trial["messages"].index(trial["m"])

                L_o_given_m = L_mixture[m_index, o_index]

                lamb_posterior *= L_o_given_m

            lamb_posteriors[lamb] = lamb_posterior

        # now renormalize lamb_posteriors to sum up to 1
        sum_posteriors = sum(lamb_posteriors.values())
        lamb_posteriors_norm = {x: lamb_posteriors[x] / sum_posteriors for x in lamb_posteriors}

        return lamb_posteriors_norm


def create_word_frequency_dict():
    word_freq_table = pd.read_csv("./data/wordFreqDict.csv").fillna(1)
    word_freq_table["Frequency"] = word_freq_table["Frequency"].astype(int)
    return word_freq_table


class RSAFrequencyModel:
    def __init__(self, alpha, priors, cost_function):
        self.word_frequency_table = create_word_frequency_dict()
        self.alpha = alpha
        self.priors = priors
        self.cost_function = cost_function

    """
    function to normalize an array by dividing each row by its sum
    to obtain a probability distribution
    """
    def normalize(self, arr):
        return arr / arr.sum(axis=1)[:, np.newaxis]

    """
    [[m]](o) -- literal meaning of utterances
    3 utterances x 3 images
    """
    def L0_truth_table(self, objects, messages):
        # t = np.zeros(shape=[len(messages), len(objects)])
        # for i in range(len(messages)):
        #     for j in range(len(objects)):
        #         # check whether the message describes object correctly
        #         if messages[i] in objects[j]:
        #             t[i, j] = 1

        t = np.array([
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])

        return t

    """
    literal listener L0(o|m)
    matrix dimensions: 3 messages x 3 objects
    """
    def L0(self, objects, messages):
        # compute [[m]](o): literal meaning
        literal_meaning = self.L0_truth_table(objects=objects, messages=messages)
        # P(o) -- create a matrix of priors by repeating the priors array for each row
        priors = np.tile(np.array(self.priors), (len(objects), 1))
        # multiply the two of L0 terms together
        unnorm = literal_meaning * priors
        # normalize to obtain a probability distribution
        norm = self.normalize(unnorm)

        return norm

    """
    pragmatic speaker S1(m|o)
    matrix dimensions: 3 images x 3 messages
    """
    def S1(self, objects, messages, beta):
        # compute the costs for each expression in messages
        message_frequencies = frequency_costs(self.word_frequency_table, messages)
        costs = self.cost_function(message_frequencies, beta=beta)

        # utility without costs
        raw_utility = np.log(self.L0(objects=objects, messages=messages).T)

        # Cost(m) -- reshape to obtain correct dimensions for matrix multiplication
        costs = np.repeat(costs, len(messages)).reshape((len(objects), len(messages))).T

        # compute utility by subtracting costs from informativity
        utility = np.subtract(raw_utility, costs)

        # compute S1(m|o) by taking the exponent of utility times the temperature parameter alpha
        unnorm = np.exp(self.alpha * utility)

        # normalize to obtain a probability distribution
        norm = self.normalize(unnorm)

        return norm

    """
    pragmatic listener L1(o|m)
    matrix dimensions: 3 messages x 3 objects
    """
    def L1(self, objects, messages, beta):
        # S1(m|o)
        S1 = self.S1(objects, messages, beta)
        # P(o) -- create a matrix of priors by repeating the priors array for each row
        priors = np.tile(np.array(self.priors), (3, 1))

        # multiply S1(m|o) (transformed, since we need a distribution over objects, not over messages) and P(o) together
        unnorm = S1.T * priors

        # normalize to obtain a probability distribution
        norm = self.normalize(unnorm)

        return norm

    def plot_L1(self, objects, messages):
        # Get the L1 values for the two messages (the first row of the L0 matrix corresponds to "beret" and the
        # second row corresponds to "scarf")
        L1_vals_weak = self.L1(objects, messages, True)[0, :]
        L1_vals_weak_nonnativ = self.L1(objects, messages, False)[0, :]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot for weak scalar, native speaker
        axes[0].bar(objects, L1_vals_weak)
        axes[0].set_title(f"hearing {messages[0]}, native speaker")
        axes[0].set_xlabel("referent")
        axes[0].set_ylabel("probability")

        # Plot for weak scalar, nonnative speaker
        axes[1].bar(objects, L1_vals_weak_nonnativ)
        axes[1].set_title(f"hearing {messages[0]}, nonnative speaker")
        axes[1].set_xlabel("referent")
        axes[1].set_ylabel("probability")

        plt.tight_layout()
        plt.show()

    def compare_L1_values(self, objects, messages):
        # Get the L1 values for the two messages (the first row of the L0 matrix corresponds to "beret" and the
        # second row corresponds to "scarf")
        L1_vals_weak = self.L1(objects, messages, beta=1)[0, :]
        L1_vals_weak_nonnativ = self.L1(objects, messages, False)[0, :]

        print(f"when hearing {messages[0]}")
        print("object    :", messages[0], messages[1])
        print("native: ", round(L1_vals_weak[0], 2), round(L1_vals_weak[1], 2))
        print("nonnative: ", round(L1_vals_weak_nonnativ[0], 2), round(L1_vals_weak_nonnativ[1], 2))

    def get_L1_values(self, objects, messages, beta):
        L1_vals = self.L1(objects, messages, beta=beta)
        return L1_vals
