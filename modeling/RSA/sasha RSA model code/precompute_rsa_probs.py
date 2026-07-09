from RSA.model_definition import *
from RSA.main import cost_function, initial_lamb_distr, mix_lambs, power_lamb_prior, compute_expected_lamb

import pickle
import numpy as np

# Data shape
N_subjects = 72  # Number of subjects
N_responses = 48  # Number of responses per subject


def compute_RSA_predictions(subject_data, alpha, favor_numeric, beta):
    predictions = []
    model = RSAUpdatingModel(priors=[0.5, 0.5], alpha=1000, cost_function=cost_function)

    # set priors
    assert (favor_numeric in [0, 1])
    favor = "low" if favor_numeric == 0 else "high"
    prior_lamb = power_lamb_prior(alpha=alpha, favor=favor)

    cur_lamb_posterior = prior_lamb
    # block order
    cur_block = 1

    for trial in subject_data:
        cur_lamb_prior = cur_lamb_posterior
        block = int(trial['block_order'])

        # if new block, mix priors
        if cur_block != block:
            cur_block = block
            cur_lamb_prior = mix_lambs(cur_lamb_posterior, prior_lamb, weight1=beta)

        l2_prob, cur_lamb_posterior = model.one_lamb_posterior_update_based_on_L2(
            lamb_priors=cur_lamb_prior,
            trial=trial,
            pragm_target=trial['pragm_target']
        )

        predictions.append(l2_prob)
    return predictions


with open('./data/subjects_72_data_dict.pkl', 'rb') as f4:
    data = pickle.load(f4)

preds = {}
alpha_grid = [1]
favor_grid = [0, 1]  # binary
beta_grid = np.linspace(0, 1, 11)
for i, alpha in enumerate(alpha_grid):
    for j, favor in enumerate(favor_grid):
        for k, beta in enumerate(beta_grid):
            print(f"alpha: {alpha}, favor: {favor}, beta: {beta}")
            predictions_grid = np.zeros(shape=(N_subjects, N_responses))
            for s in range(N_subjects):
                predictions_grid[s, :] = compute_RSA_predictions(
                    subject_data=data[s],
                    alpha=alpha,
                    favor_numeric=favor,
                    beta=beta
                )

            preds[(int(alpha), int(favor), round(float(beta), 2))] = predictions_grid

print(preds)

# with open("./data/predictions_72.pkl", "rb") as f1:
#     all_predictions = pickle.load(f1)
#
# all_predictions.update(preds)
#
# with open("./data/predictions_complete_72.pkl", "wb") as f2:
#     pickle.dump(all_predictions, f2)
