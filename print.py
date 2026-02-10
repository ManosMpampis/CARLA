import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import scale
from matplotlib import ticker

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def rule(experiment, graph, df):
    if experiment == "original":
        if graph == "p_loss":
            rule = df['Step'] > 0
        elif graph == "n_loss":
            rule = df['Step'] > 0
        elif graph == "Margin":
            rule = df['Step'] > 5
        elif graph == "Full Loss":
            rule = df['Step'] > 0
        else:
            rule = df['Step'] > 5
    else:
        if graph == "p_loss":
            rule = df['Step'] > 5
        elif graph == "n_loss":
            rule = df['Step'] > 10
        elif graph == "Margin":
            rule = df['Step'] > 5
        elif graph == "Full Loss":
            rule = df['Step'] > 5
        else:
            rule = df['Step'] > 5
    return rule

# def rule(experiment, graph, df):
#     if experiment == "original":
#         if graph == "consistency":
#             rule = df['Step'] >= 0
#         elif graph == "incon":
#             rule = df['Step'] >= 0
#         elif graph == "entropy":
#             rule = df['Step'] >= 0
#         elif graph == "total":
#             rule = df['Step'] >= 0
#         else:
#             rule = df['Step'] >= 0
#     elif experiment == "experiment":
#         if graph == "consistency":
#             rule = df['Step'] >= 0
#         elif graph == "incon":
#             rule = df['Step'] >= 0
#         elif graph == "entropy":
#             rule = df['Step'] >= 0
#         elif graph == "total":
#             rule = df['Step'] >= 0
#         else:
#             rule = df['Step'] >= 0
#     elif experiment == "experiment norm entropy":
#         if graph == "consistency":
#             rule = df['Step'] >= 5
#         elif graph == "incon":
#             rule = df['Step'] >= 5
#         elif graph == "entropy":
#             rule = df['Step'] >= 5
#         elif graph == "total":
#             rule = df['Step'] >= 5
#         else:
#             rule = df['Step'] >= 0
#     return rule

def main():
    colors = ["darkorange", "lightcoral", "lawngreen", "cyan", "plum"]
    # experiments = ["original","experiment", "experiment norm entropy"]
    # graphs = ["consistency", "incon", "entropy", "total"]
    # labels = ["Consistency Loss", "Inconsistency Loss", "Entropy", "Total Loss"]
    experiments = ["original","experiment"]
    graphs = ["Margin"]#["p_loss", "n_loss", "Margin", "Full Loss"]
    labels = ["Margin"]#["Positive Loss", "Negative Loss", "Margin", "Total Loss"]
    
    for experiment in experiments:
        for graph, Label in zip(graphs, labels):
            dir = f'/home/manos/Downloads/pretext/{experiment}/{graph}'
            # dir = f'/home/manos/Downloads/self_sup/{experiment}/{graph}'

            data_info = os.listdir(dir)
            files = [file for file in data_info if file.endswith('.csv')]
            files = sorted(files)
            for filename in files:
                df = pd.read_csv(f"{dir}/{filename}")

                r= rule(experiment, graph, df)
                x = df['Step'][r]
                y = df['Value'][r]
                
                plt.figure(figsize=(5, 5))
                plt.plot(x, y, color="darkorange")

                plt.xlabel("Epoches")
                plt.yscale("logit")
                plt.ylabel(f"{Label}")
                plt.savefig(f'{dir}/{filename.split(".")[0]}.png', bbox_inches='tight')
            
            plt.figure(figsize=(5, 5))
            for i, filename in enumerate(files):
                df = pd.read_csv(f"{dir}/{filename}")
                
                r = rule(experiment, graph, df)
                x = df['Step'][r]
                # y = np.log10(df['Value'][r])
                y = df['Value'][r]

                plt.plot(x, y, color=colors[i], label=filename.split(".")[0])
                plt.xlabel("Epoches")
                # plt.yscale("symlog")
                plt.ylabel(f"{Label}")

            plt.legend(loc="upper right")
            plt.savefig(f'{dir}/{Label}.png', bbox_inches='tight')


if __name__ == "__main__":
    main()

