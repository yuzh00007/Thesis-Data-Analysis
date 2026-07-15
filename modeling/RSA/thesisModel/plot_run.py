import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from helper import (
    cost_function_log_frequency,
    cost_function_inverse_log_frequency,
    cost_function_inverse_cube_root,
)


def generate_beta_cost_graphs(cost_function, word_freq_df, beta_list):
    """
    map frequency distribution for different beta values

    graph x-axis: word frequency
    graph y-axis: cost of uttering word
    """
    costs = {}
    for beta in beta_list:
        # beta_costs = [math.log(freq_max * 2 / (beta + freq)) for beta in beta_values]
        beta_costs = cost_function(word_freq_df["Frequency"], beta)
        costs.update({
            beta: beta_costs
        })

    plt.figure()
    for beta, beta_costs in costs.items():
        plt.scatter(word_freq_df["Frequency"], beta_costs, marker='o', label=beta)

    plt.xlabel("Frequency")
    plt.ylabel("Cost")
    plt.title("Freq-Cost at Different Betas")
    plt.legend()
    plt.draw()
    plt.savefig(f"./output/FreqCostChart{cost_function.__name__.replace("cost_function", "")}.png")

    return


if __name__ == "__main__":
    # min value is 431, give NAs a value slightly lower (Zipf's law)
    # hardcoded the wordFreqDict so they have rank 20000 and frequency 400
    all_words = pd.read_csv("./data/wordFreqDict.csv").fillna(400)
    all_words = all_words[["Word", "Frequency"]]
    freq_max = all_words["Frequency"].max()

    # #########################
    # # Generating BETA COSTS #
    # #########################
    # cost_functions = {
    #     cost_function_log_frequency: [.05, .1, .5, 1, 1.2, 1.4, 1.6, 2],
    #     cost_function_inverse_log_frequency: [500000, 100000, 20000, 4000, 800, 125],
    #     cost_function_inverse_cube_root: [400000, 40000, 4000, 400, 40, 4, 1, .1, .01, .001]
    # }
    #
    # for costfunction, betas in cost_functions.items():
    #     generate_beta_cost_graphs(costfunction, all_words, betas)

    ###################################
    # Generating PREDICTION vs. HUMAN #
    ###################################
    # itemid  freq_weak  freq_strong  derivationNative  derivationNotNative
    model_predictions = pd.read_csv("./output/predictions.csv")
    # itemId speakerType  derivationStrength
    human_data = pd.read_csv("./data/derivationByItemSpeaker.csv")

    # PLOT
    plt.figure()
    plt.scatter(
        model_predictions[
            model_predictions["itemid"] != 13
        ]["derivationNotNative"][:15],
        human_data[
            (human_data["speakerType"] != "native")
            & (human_data["itemId"] != 13)
        ]["avgDerivationStrength"][:15],
        marker='o',
        label="nonnative",
        color="steelblue"
    )
    x_not_nat = model_predictions[
            model_predictions["itemid"] != 13
        ]["derivationNotNative"][:15]
    y_not_nat = human_data[
            (human_data["speakerType"] != "native")
            & (human_data["itemId"] != 13)
        ]["avgDerivationStrength"][:15]

    a, b = np.polyfit(x_not_nat, y_not_nat, 1)
    plt.plot(x_not_nat, a * x_not_nat + b, color="steelblue", label=f"y = {round(a, 2)}x + {round(b, 2)}")

    plt.scatter(
        model_predictions[
            model_predictions["itemid"] != 13
        ]["derivationNative"][:15],
        human_data[
            (human_data["speakerType"] == "native")
            & (human_data["itemId"] != 13)  # face one, outlier
        ]["avgDerivationStrength"][:15],
        marker='x',
        label="Native",
        color="orange"
    )
    x_nativ = model_predictions[
        model_predictions["itemid"] != 13
        ]["derivationNative"][:15]
    y_nativ = human_data[
        (human_data["speakerType"] == "native")
        & (human_data["itemId"] != 13)  # face one, outlier
    ]["avgDerivationStrength"][:15]

    a, b = np.polyfit(x_nativ, y_nativ, 1)
    plt.plot(x_nativ, a * x_nativ + b, color="orange", label=f"y = {round(a, 2)}x + {round(b, 2)}")

    plt.xlabel("Model")
    plt.ylabel("Human")
    plt.legend()
    plt.title("Critical Only Model vs Human Predictions")
    plt.savefig("./output/prediction-human.png")


    print(
        pd.DataFrame(model_predictions[
             model_predictions["itemid"] != 13
             ]["derivationNative"]
        ).corrwith(
            human_data[
                (human_data["speakerType"] == "native")
                & (human_data["itemId"] != 13)  # face one, outlier
            ]["avgDerivationStrength"]
        )
    )

    print(
        pd.DataFrame(model_predictions[
             model_predictions["itemid"] != 13
             ]["derivationNotNative"]
         ).corrwith(
            human_data[
                (human_data["speakerType"] != "native")
                & (human_data["itemId"] != 13)  # face one, outlier
            ]["avgDerivationStrength"]
        )
    )

    print(
        pd.DataFrame(
            human_data[
                (human_data["speakerType"] != "native")
                & (human_data["itemId"] != 13)  # face one, outlier
            ]["avgDerivationStrength"]
        ).corrwith(
            model_predictions[
                model_predictions["itemid"] != 13
            ]["derivationNotNative"]
        )
    )
