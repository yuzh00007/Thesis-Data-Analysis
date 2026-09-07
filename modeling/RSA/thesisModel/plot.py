import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from helper import cost_function_inverse_cube_root


def generate_human_v_model_plot():
    for allCrit in [True, False, None]:
        if allCrit is None:
            file_ext = "allTrials"
            title = "All Items"
        elif allCrit is True:
            file_ext = "allCrit"
            title = "Critical Items"
        else:
            file_ext = "effectOnly"
            title = "Only Effect Items"

        df_native = pd.read_csv(f"output/native/humanVsModel{file_ext}.csv")
        df_nonnat = pd.read_csv(f"output/nonnat/humanVsModel{file_ext}.csv")
        df = pd.concat([df_nonnat, df_native], ignore_index=True)

        if allCrit in [True, False]:
            g = sns.scatterplot(
                data=df,
                y="p(weak|weak)", x="derivationStrength",
                alpha=.6, hue="speakerType"
            )
        else:
            g = sns.scatterplot(
                x="derivationStrength", y="p(weak|weak)",
                hue="speakerType", style="itemType", data=df
            )

        g.set(
            xlabel="Avg. Human Derivation Strength",
            ylabel="P(Weak-Target | Weak-Scalar)"
        )

        plt.savefig(f"./output/plots/humanVsPrediction{file_ext}.png")
        plt.clf()


def generate_cost_curve():
    # nonnat_beta = .00001
    # nat_beta = .00019

    betas = [.00000054, .000007, .00001, .125]

    # nonnat = cost_function_inverse_cube_root(range(1, 10000), nonnat_beta)
    # nat = cost_function_inverse_cube_root(range(1, 10000), nat_beta)

    costs = {}
    for beta in betas:
        costs.update({
            beta: cost_function_inverse_cube_root(range(1, 10000), beta)
        })

    df = pd.DataFrame(costs).reset_index()
    custom_palette = sns.color_palette(["#035c00", "#ff1cd2", "#0071ac", "#c85317"])

    dfm = df.melt('index', var_name='Speaker Type', value_name='vals')
    g = sns.lineplot(data=dfm, x="index", y="vals", hue="Speaker Type", alpha=.8, palette=custom_palette)

    g.set(
        xlabel="Word Frequency",
        ylabel="Cost"
    )

    plt.savefig("./output/plots/betaCostCurves.png")


if __name__ == "__main__":
    generate_cost_curve()
