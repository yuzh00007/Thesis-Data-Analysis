from helper import frequency_costs, create_word_frequency_dict

import numpy as np
import matplotlib.pyplot as plt


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
