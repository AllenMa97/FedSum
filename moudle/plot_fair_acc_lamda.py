import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl

def origin():
    def ADULT_LR(path, show_figure=False):
        λ_list = [0.125, 0.25, 0.5, 1, 2, 4]

        acc = [0.8321, 0.8326, 0.8474, 0.8065, 0.7839, 0.7638]
        FR = [0.6606, 0.666, 0.8269, 0.8543, 0.9069, 1.0]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        fig = plt.figure(figsize=(7, 8), dpi=100)
        plt.subplots_adjust(hspace=0.16)

        ax = fig.add_subplot(211)

        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='red', linewidth=2)
        plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=10)

        ax.set_ylabel("Fairness", size=11, color="red")
        plt.tick_params(axis='y', colors='red', labelsize=10)
        plt.legend(bbox_to_anchor=(0.25, 1.14), borderaxespad=0, fontsize=12, frameon=False)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="blue", label="Uniform Distribution", linestyle="dashed",
                        linewidth=2)
        ax1.set_ylabel("Accuracy", size=11, color='blue')
        plt.tick_params(axis='y', colors='blue', labelsize=10)
        plt.legend(bbox_to_anchor=(1.00, 1.14), borderaxespad=0, fontsize=12, frameon=False)

        acc = [0.8573, 0.8568, 0.8493, 0.8469, 0.8483, 0.7652]
        FR = [0.8636, 0.8865, 0.8882, 0.9764, 0.9254, 0.9946]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        ax = fig.add_subplot(212)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='red', linewidth=2)
        ax.set_xlabel("λ", size=10)
        plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], rotation=0)

        ax.set_ylabel("Fairness", size=11, color="red")
        plt.tick_params(axis='y', colors='red', labelsize=10)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="green", label="Uniform Client", linestyle="dashed", linewidth=2)
        ax1.set_ylabel("Accuracy", size=11, color='green')
        plt.tick_params(axis='y', colors='green', labelsize=10)
        plt.legend(bbox_to_anchor=(0.58, 2.24), borderaxespad=0, fontsize=12, frameon=False)

        plt.savefig(path, show_figure=False)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        if show_figure:
            plt.show()

    def ADULT_NN(path, show_figure=False):
        λ_list = [0.125, 0.25, 0.5, 1, 2, 4]

        acc = [0.8111, 0.823, 0.8171, 0.7841, 0.7837, 0.7638]
        FR = [0.6868, 0.7443, 0.7732, 0.883, 0.8857, 1.0]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        fig = plt.figure(figsize=(7, 8), dpi=100)
        plt.subplots_adjust(hspace=0.16)

        ax = fig.add_subplot(211)

        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='red', linewidth=2)
        plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=10)

        ax.set_ylabel("Fairness", size=11, color="red")
        plt.tick_params(axis='y', colors='red', labelsize=10)
        plt.legend(bbox_to_anchor=(0.25, 1.14), borderaxespad=0, fontsize=12, frameon=False)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="blue", label="Uniform Distribution", linestyle="dashed",
                        linewidth=2)
        ax1.set_ylabel("Accuracy", size=11, color='blue')
        plt.tick_params(axis='y', colors='blue', labelsize=10)
        plt.legend(bbox_to_anchor=(1.00, 1.14), borderaxespad=0, fontsize=12, frameon=False)

        acc = [0.8616, 0.8642, 0.8618, 0.8655, 0.7638, 0.7638]
        FR = [0.8628, 0.9044, 0.8915, 0.9303, 1, 1]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        ax = fig.add_subplot(212)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='red', linewidth=2)
        ax.set_xlabel("λ", size=10)
        plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], rotation=0)

        ax.set_ylabel("Fairness", size=11, color="red")
        plt.tick_params(axis='y', colors='red', labelsize=10)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="green", label="Uniform Client", linestyle="dashed", linewidth=2)
        ax1.set_ylabel("Accuracy", size=11, color='green')
        plt.tick_params(axis='y', colors='green', labelsize=10)
        plt.legend(bbox_to_anchor=(0.58, 2.24), borderaxespad=0, fontsize=12, frameon=False)

        plt.savefig(path, show_figure=False)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        if show_figure:
            plt.show()

    def COMPAS_LR(path, show_figure=False):
        λ_list = [0.125, 0.25, 0.5, 1, 2, 4]

        acc = [0.7008, 0.7071, 0.705, 0.7008, 0.7029, 0.6862]
        FR = [0.6265, 0.6538, 0.6983, 0.7034, 0.7214, 0.765]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        fig = plt.figure(figsize=(7, 8), dpi=100)
        plt.subplots_adjust(hspace=0.16)

        ax = fig.add_subplot(211)

        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='red', linewidth=2)
        plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=10)

        ax.set_ylabel("Fairness", size=11, color="red")
        plt.tick_params(axis='y', colors='red', labelsize=10)
        plt.legend(bbox_to_anchor=(0.25, 1.14), borderaxespad=0, fontsize=12, frameon=False)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="blue", label="Uniform Distribution", linestyle="dashed",
                        linewidth=2)
        ax1.set_ylabel("Accuracy", size=11, color='blue')
        plt.tick_params(axis='y', colors='blue', labelsize=10)
        plt.legend(bbox_to_anchor=(1.00, 1.14), borderaxespad=0, fontsize=12, frameon=False)

        acc = [0.7071, 0.7113, 0.6904, 0.7008, 0.6967, 0.6904]
        FR = [0.6316, 0.6256, 0.6393, 0.6444, 0.6615, 0.6684]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        ax = fig.add_subplot(212)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='red', linewidth=2)
        ax.set_xlabel("λ", size=10)
        plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], rotation=0)

        ax.set_ylabel("Fairness", size=11, color="red")
        plt.tick_params(axis='y', colors='red', labelsize=10)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="green", label="Uniform Client", linestyle="dashed", linewidth=2)
        ax1.set_ylabel("Accuracy", size=11, color='green')
        plt.tick_params(axis='y', colors='green', labelsize=10)
        plt.legend(bbox_to_anchor=(0.58, 2.24), borderaxespad=0, fontsize=12, frameon=False)

        plt.savefig(path, show_figure=False)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        if show_figure:
            plt.show()

    def COMPAS_NN(path, show_figure=False):
        λ_list = [0.125, 0.25, 0.5, 1, 2, 4]

        acc = [0.6987, 0.7092, 0.705, 0.7029, 0.4833, 0.4833]
        FR = [0.6497, 0.6547, 0.6718, 0.7291, 1, 1]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        fig = plt.figure(figsize=(7, 8), dpi=100)
        plt.subplots_adjust(hspace=0.16)

        ax = fig.add_subplot(211)

        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='red', linewidth=2)
        plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=10)

        ax.set_ylabel("Fairness", size=11, color="red")
        plt.tick_params(axis='y', colors='red', labelsize=10)
        plt.legend(bbox_to_anchor=(0.25, 1.14), borderaxespad=0, fontsize=12, frameon=False)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="blue", label="Uniform Distribution", linestyle="dashed",
                        linewidth=2)
        ax1.set_ylabel("Accuracy", size=11, color='blue')
        plt.tick_params(axis='y', colors='blue', labelsize=10)
        plt.legend(bbox_to_anchor=(1.00, 1.14), borderaxespad=0, fontsize=12, frameon=False)

        acc = [0.6904, 0.6695, 0.5921, 0.5167, 0.5167, 0.5167]
        FR = [0.6897, 0.5923, 0.6872, 1, 1, 1]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        ax = fig.add_subplot(212)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='red', linewidth=2)
        ax.set_xlabel("λ", size=10)
        plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], rotation=0)

        ax.set_ylabel("Fairness", size=11, color="red")
        plt.tick_params(axis='y', colors='red', labelsize=10)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="green", label="Uniform Client", linestyle="dashed", linewidth=2)
        ax1.set_ylabel("Accuracy", size=11, color='green')
        plt.tick_params(axis='y', colors='green', labelsize=10)
        plt.legend(bbox_to_anchor=(0.58, 2.24), borderaxespad=0, fontsize=12, frameon=False)

        plt.savefig(path, show_figure=False)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        if show_figure:
            plt.show()

    def DRUG_LR(path, show_figure=False):
        λ_list = [0.125, 0.25, 0.5, 1, 2, 4]

        acc = [0.7817, 0.7852, 0.7923, 0.7817, 0.7958, 0.7676]
        FR = [0.8887, 0.9133, 0.9209, 0.9307, 0.98, 1]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        fig = plt.figure(figsize=(7, 8), dpi=100)
        plt.subplots_adjust(hspace=0.16)

        ax = fig.add_subplot(211)

        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='red', linewidth=2)
        plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=10)

        ax.set_ylabel("Fairness", size=11, color="red")
        plt.tick_params(axis='y', colors='red', labelsize=10)
        plt.legend(bbox_to_anchor=(0.25, 1.14), borderaxespad=0, fontsize=12, frameon=False)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="blue", label="Uniform Distribution", linestyle="dashed",
                        linewidth=2)
        ax1.set_ylabel("Accuracy", size=11, color='blue')
        plt.tick_params(axis='y', colors='blue', labelsize=10)
        plt.legend(bbox_to_anchor=(1.00, 1.14), borderaxespad=0, fontsize=12, frameon=False)

        acc = [0.7852, 0.7782, 0.7852, 0.7746, 0.7711, 0.7746]
        FR = [0.9455, 0.9652, 0.9951, 1, 1, 1]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        ax = fig.add_subplot(212)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='red', linewidth=2)
        ax.set_xlabel("λ", size=10)
        plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], rotation=0)

        ax.set_ylabel("Fairness", size=11, color="red")
        plt.tick_params(axis='y', colors='red', labelsize=10)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="green", label="Uniform Client", linestyle="dashed", linewidth=2)
        ax1.set_ylabel("Accuracy", size=11, color='green')
        plt.tick_params(axis='y', colors='green', labelsize=10)
        plt.legend(bbox_to_anchor=(0.58, 2.24), borderaxespad=0, fontsize=12, frameon=False)

        plt.savefig(path, show_figure=False)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        if show_figure:
            plt.show()

    def DRUG_NN(path, show_figure=False):
        λ_list = [0.125, 0.25, 0.5, 1, 2, 4]

        acc = [0.7676, 0.7676, 0.7923, 0.7676, 0.7676, 0.7923]
        FR = [1, 1, 0.9455, 1, 1, 0.9455]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        fig = plt.figure(figsize=(7, 8), dpi=100)
        plt.subplots_adjust(hspace=0.16)

        ax = fig.add_subplot(211)

        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='red', linewidth=2)
        plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=10)

        ax.set_ylabel("Fairness", size=11, color="red")
        plt.tick_params(axis='y', colors='red', labelsize=10)
        plt.legend(bbox_to_anchor=(0.25, 1.14), borderaxespad=0, fontsize=12, frameon=False)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="blue", label="Uniform Distribution", linestyle="dashed",
                        linewidth=2)
        ax1.set_ylabel("Accuracy", size=11, color='blue')
        plt.tick_params(axis='y', colors='blue', labelsize=10)
        plt.legend(bbox_to_anchor=(1.00, 1.14), borderaxespad=0, fontsize=12, frameon=False)

        acc = [0.7676, 0.7676, 0.7676, 0.7676, 0.7676, 0.7676]
        FR = [1, 1, 1, 1, 1, 1]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        ax = fig.add_subplot(212)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='red', linewidth=2)
        ax.set_xlabel("λ", size=10)
        plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], rotation=0)

        ax.set_ylabel("Fairness", size=11, color="red")
        plt.tick_params(axis='y', colors='red', labelsize=10)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="green", label="Uniform Client", linestyle="dashed", linewidth=2)
        ax1.set_ylabel("Accuracy", size=11, color='green')
        plt.tick_params(axis='y', colors='green', labelsize=10)
        plt.legend(bbox_to_anchor=(0.58, 2.24), borderaxespad=0, fontsize=12, frameon=False)

        plt.savefig(path, show_figure=False)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        if show_figure:
            plt.show()

    path = "../save_path/figure/ADULT/LR/fair_acc_lamda.png"
    ADULT_LR(path, show_figure=False)
    path = "../save_path/figure/ADULT/NN/fair_acc_lamda.png"
    ADULT_NN(path, show_figure=False)

    path = "../save_path/figure/COMPAS/LR/fair_acc_lamda.png"
    COMPAS_LR(path, show_figure=False)
    path = "../save_path/figure/COMPAS/NN/fair_acc_lamda.png"
    COMPAS_NN(path, show_figure=False)

    path = "../save_path/figure/DRUG/LR/fair_acc_lamda.png"
    DRUG_LR(path, show_figure=False)
    path = "../save_path/figure/DRUG/NN/fair_acc_lamda.png"
    DRUG_NN(path, show_figure=False)

def DRUG():
    def LR_Data_Partition_Dirichlet(path, show_figure=False):
        λ_list = ["0.125", "0.25", "0.5", "1", "2", "4"]
        acc = [0.7958,0.7852,0.7923,0.7923,0.7958,0.7676]
        FR = [0.9924, 0.9133, 0.9209, 0.9455, 0.9878, 1]
        acc = [round(item,2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        fig = plt.figure(figsize=(6, 6), dpi=100)
        plt.subplots_adjust(hspace=0.16)

        ax = fig.add_subplot(211)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=10)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        # ax.set_ylabel("Fairness", size=12)
        plt.legend(bbox_to_anchor=(0.15, 1.14), borderaxespad=0, fontsize=10, frameon=False)
        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="blue", label="FedRényi Accuracy (for \n Uniform Over Distribution)", linestyle="dashed", linewidth=2)
        plt.yticks([0.77, 0.78, 0.79, 0.80], rotation=90)
        plt.tick_params(axis='y', labelsize=10)
        plt.legend(bbox_to_anchor=(0.66, 1.18), borderaxespad=0, fontsize=10, frameon=False)

        acc = [0.7817, 0.7852, 0.7923, 0.7817, 0.7958, 0.7676]
        FR = [0.8887,0.9133,0.9209,0.9307,0.98,1]
        acc = [round(item, 2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        ax = fig.add_subplot(212)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        ax.set_xlabel("λ (DRUG, Logistic Regression)", size=14)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)

        ax.set_ylabel("                                              Fairness", size=12)
        plt.tick_params(axis='y', labelsize=10, rotation=90)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="green", label="FedRényi Accuracy (for \n Uniform Over Client)", linestyle="dotted", linewidth=2)
        ax1.set_ylabel("                                                     Accuracy (Data Partition: Dirichlet)", size=12)
        plt.yticks([0.77, 0.78, 0.79, 0.80], rotation=90)
        plt.tick_params(axis='y', labelsize=10)
        plt.legend(bbox_to_anchor=(1.11, 2.34), borderaxespad=0, fontsize=10, frameon=False)
        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")

        if show_figure:
            plt.show()

    def LR_Data_Partition_Uniform(path, show_figure=False):
        λ_list = ["0.125", "0.25", "0.5", "1", "2", "4"]
        acc = [0.7852,0.7782,0.7852,0.7746,0.7711,0.7746]
        FR = [0.9455,0.9652,0.9951,1,1,1]
        acc = [round(item, 2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        fig = plt.figure(figsize=(6, 6), dpi=100)
        plt.subplots_adjust(hspace=0.16)

        ax = fig.add_subplot(211)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=10)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        # ax.set_ylabel("Fairness", size=12)
        plt.legend(bbox_to_anchor=(0.15, 1.14), borderaxespad=0, fontsize=10, frameon=False)
        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="blue", label="FedRényi Accuracy (for \n Uniform Over Distribution)",
                        linestyle="dashed", linewidth=2)
        plt.yticks([0.77, 0.78, 0.79, 0.80], rotation=90)
        plt.tick_params(axis='y', labelsize=10)
        plt.legend(bbox_to_anchor=(0.66, 1.18), borderaxespad=0, fontsize=10, frameon=False)

        acc = [0.7852, 0.7782 , 0.7817 , 0.7676 , 0.7676, 0.7676]
        FR = [0.9455, 0.9652, 0.9901, 1, 1, 1]
        acc = [round(item, 2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        ax = fig.add_subplot(212)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        ax.set_xlabel("λ (DRUG, Logistic Regression)", size=14)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)

        ax.set_ylabel("                                              Fairness", size=12)
        plt.tick_params(axis='y', labelsize=10, rotation=90)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="green", label="FedRényi Accuracy (for \n Uniform Over Client)",
                        linestyle="dotted", linewidth=2)
        ax1.set_ylabel("                                                     Accuracy (Data Partition: Uniform)",
                       size=12)
        plt.yticks([0.77, 0.78, 0.79, 0.80], rotation=90)
        plt.tick_params(axis='y', labelsize=10)
        plt.legend(bbox_to_anchor=(1.11, 2.34), borderaxespad=0, fontsize=10, frameon=False)
        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")

        if show_figure:
            plt.show()

    def NN_Data_Partition_Dirichlet(path, show_figure=False):
        λ_list = ["0.125", "0.25", "0.5", "1", "2", "4"]
        acc = [0.7676,0.7676,0.7676,0.7676,0.7676,0.7676]
        FR = [1, 1, 1, 1, 1, 1]
        acc = [round(item, 2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        fig = plt.figure(figsize=(6, 6), dpi=100)
        plt.subplots_adjust(hspace=0.16)

        ax = fig.add_subplot(211)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=10)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        # ax.set_ylabel("Fairness", size=12)
        plt.legend(bbox_to_anchor=(0.15, 1.14), borderaxespad=0, fontsize=10, frameon=False)
        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="blue", label="FedRényi Accuracy (for \n Uniform Over Distribution)",
                        linestyle="dashed", linewidth=2)
        # ax1.set_ylabel("Accuracy (Data Partition: Dirichlet)", size=12)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        plt.legend(bbox_to_anchor=(0.66, 1.18), borderaxespad=0, fontsize=10, frameon=False)

        acc = [0.7676,0.7676,0.7923,0.7676,0.7676,0.7923]
        FR = [1,1,0.9455,1,1,0.9455]
        acc = [round(item, 2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        ax = fig.add_subplot(212)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        ax.set_xlabel("λ (DRUG, Neural Network)", size=14)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)

        ax.set_ylabel("                                              Fairness", size=12)
        plt.tick_params(axis='y', labelsize=10, rotation=90)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="green", label="FedRényi Accuracy(for \n Uniform Over Client)",
                        linestyle="dotted", linewidth=2)
        ax1.set_ylabel("                                                     Accuracy (Data Partition: Dirichlet)",
                       size=12)
        plt.yticks([0.77, 0.78, 0.79], rotation=90)
        plt.tick_params(axis='y', labelsize=10)
        plt.legend(bbox_to_anchor=(1.11, 2.34), borderaxespad=0, fontsize=10, frameon=False)
        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")

        if show_figure:
            plt.show()

    def NN_Data_Partition_Uniform(path, show_figure=False):
        λ_list = ["0.125", "0.25", "0.5", "1", "2", "4"]
        acc = [0.7676,0.7676,0.7676,0.7676,0.7676,0.7676]
        FR = [1, 1, 1, 1, 1, 1]
        acc = [round(item, 2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        fig = plt.figure(figsize=(6, 6), dpi=100)
        plt.subplots_adjust(hspace=0.16)

        ax = fig.add_subplot(211)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=10)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        # ax.set_ylabel("Fairness", size=12)
        plt.legend(bbox_to_anchor=(0.15, 1.14), borderaxespad=0, fontsize=10, frameon=False)
        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="blue", label="FedRényi Accuracy (for \n Uniform Over Distribution)",
                        linestyle="dashed", linewidth=2)
        # ax1.set_ylabel("Accuracy (Data Partition: Dirichlet)", size=12)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        plt.legend(bbox_to_anchor=(0.66, 1.18), borderaxespad=0, fontsize=10, frameon=False)

        acc = [0.7676,0.7676,0.7676,0.7676,0.7676,0.7711]
        FR = [1, 1, 1, 1, 1, 1]
        acc = [round(item, 2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        ax = fig.add_subplot(212)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        ax.set_xlabel("λ (DRUG, Neural Network)", size=14)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)

        ax.set_ylabel("                                              Fairness", size=12)
        plt.tick_params(axis='y', labelsize=10, rotation=90)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="green", label="FedRényi Accuracy(for \n Uniform Over Client)",
                        linestyle="dotted", linewidth=2)
        ax1.set_ylabel("                                                     Accuracy (Data Partition: Uniform)",
                       size=12)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        plt.legend(bbox_to_anchor=(1.11, 2.34), borderaxespad=0, fontsize=10, frameon=False)
        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")

        if show_figure:
            plt.show()


    path = "../save_path/figure/DRUG/LR/Lamda_Data_Partition_Dirichlet.png"
    LR_Data_Partition_Dirichlet(path, show_figure=False)
    path = "../save_path/figure/DRUG/LR/Lamda_Data_Partition_Uniform.png"
    LR_Data_Partition_Uniform(path, show_figure=False)
    path = "../save_path/figure/DRUG/NN/Lamda_Data_Partition_Dirichlet.png"
    NN_Data_Partition_Dirichlet(path, show_figure=False)
    path = "../save_path/figure/DRUG/NN/Lamda_Data_Partition_Uniform.png"
    NN_Data_Partition_Uniform(path, show_figure=False)

def COMPAS():
    def LR_Data_Partition_Dirichlet(path, show_figure=False):
        λ_list = ["0.125", "0.25", "0.5", "1", "2", "4"]
        acc = [0.705, 0.7071, 0.6925, 0.6987, 0.7029, 0.6862]
        FR = [0.6538, 0.6538, 0.7085, 0.7179, 0.7231, 0.765]
        acc = [round(item,2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        fig = plt.figure(figsize=(6, 6), dpi=100)
        plt.subplots_adjust(hspace=0.16)

        ax = fig.add_subplot(211)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=10)
        plt.yticks([0.65, 0.69, 0.73, 0.77], rotation=90)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        # ax.set_ylabel("Fairness", size=12)
        plt.legend(bbox_to_anchor=(0.15, 1.14), borderaxespad=0, fontsize=10, frameon=False)
        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="blue", label="FedRényi Accuracy (for \n Uniform Over Distribution)", linestyle="dashed", linewidth=2)
        plt.yticks([0.69, 0.70, 0.71], rotation=90)
        plt.tick_params(axis='y', labelsize=10)
        plt.legend(bbox_to_anchor=(0.66, 1.18), borderaxespad=0, fontsize=10, frameon=False)

        acc = [0.7008, 0.7071, 0.7005, 0.7008, 0.7029, 0.6862]
        FR = [0.6265, 0.6538, 0.6983, 0.7034, 0.7214, 0.765]
        acc = [round(item, 2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        ax = fig.add_subplot(212)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        ax.set_xlabel("λ (COMPAS, Logistic Regression)", size=14)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)

        ax.set_ylabel("                                              Fairness", size=12)
        plt.yticks([0.62, 0.68, 0.73, 0.78], rotation=90)
        plt.tick_params(axis='y', labelsize=10, rotation=90)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="green", label="FedRényi Accuracy (for \n Uniform Over Client)", linestyle="dotted", linewidth=2)
        ax1.set_ylabel("                                                     Accuracy (Data Partition: Dirichlet)", size=12)
        plt.yticks([0.69, 0.70, 0.71], rotation=90)
        plt.tick_params(axis='y', labelsize=10)
        plt.legend(bbox_to_anchor=(1.11, 2.34), borderaxespad=0, fontsize=10, frameon=False)
        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")

        if show_figure:
            plt.show()

    def LR_Data_Partition_Uniform(path, show_figure=False):
        λ_list = ["0.125", "0.25", "0.5", "1", "2", "4"]
        acc = [0.7071, 0.7113, 0.6904, 0.7008, 0.6967, 0.6904]
        FR = [0.6316, 0.6256, 0.6393, 0.6444, 0.6615, 0.6684]
        acc = [round(item, 2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        fig = plt.figure(figsize=(6, 6), dpi=100)
        plt.subplots_adjust(hspace=0.16)

        ax = fig.add_subplot(211)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=10)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        # ax.set_ylabel("Fairness", size=12)
        plt.legend(bbox_to_anchor=(0.15, 1.14), borderaxespad=0, fontsize=10, frameon=False)
        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="blue", label="FedRényi Accuracy (for \n Uniform Over Distribution)",
                        linestyle="dashed", linewidth=2)
        plt.yticks([0.69, 0.70, 0.71], rotation=90)
        plt.tick_params(axis='y', labelsize=10)
        plt.legend(bbox_to_anchor=(0.66, 1.18), borderaxespad=0, fontsize=10, frameon=False)

        acc = [0.7071, 0.7113, 0.6883, 0.6883, 0.7008, 0.705]
        FR = [0.6316, 0.6256, 0.6761, 0.6675, 0.6718, 0.6684]
        acc = [round(item, 2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        ax = fig.add_subplot(212)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        ax.set_xlabel("λ (COMPAS, Logistic Regression)", size=14)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)

        ax.set_ylabel("                                              Fairness", size=12)
        plt.tick_params(axis='y', labelsize=10, rotation=90)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="green", label="FedRényi Accuracy (for \n Uniform Over Client)",
                        linestyle="dotted", linewidth=2)
        ax1.set_ylabel("                                                     Accuracy (Data Partition: Uniform)",
                       size=12)
        plt.yticks([0.69, 0.70, 0.71], rotation=90)
        plt.tick_params(axis='y', labelsize=10)
        plt.legend(bbox_to_anchor=(1.11, 2.34), borderaxespad=0, fontsize=10, frameon=False)
        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")

        if show_figure:
            plt.show()

    def NN_Data_Partition_Dirichlet(path, show_figure=False):
        λ_list = ["0.125", "0.25", "0.5", "1", "2", "4"]
        acc = [0.7092, 0.705, 0.705, 0.5167, 0.4833, 0.4833]
        FR = [0.6709, 0.6726, 0.741, 1, 1, 1]
        acc = [round(item, 2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        fig = plt.figure(figsize=(6, 6), dpi=100)
        plt.subplots_adjust(hspace=0.16)

        ax = fig.add_subplot(211)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=10)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        # ax.set_ylabel("Fairness", size=12)
        plt.legend(bbox_to_anchor=(0.15, 1.14), borderaxespad=0, fontsize=10, frameon=False)
        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="blue", label="FedRényi Accuracy (for \n Uniform Over Distribution)",
                        linestyle="dashed", linewidth=2)
        # ax1.set_ylabel("Accuracy (Data Partition: Dirichlet)", size=12)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        plt.legend(bbox_to_anchor=(0.66, 1.18), borderaxespad=0, fontsize=10, frameon=False)

        acc = [0.6987, 0.7092, 0.705, 0.7029, 0.4833, 0.4833]
        FR = [0.6497, 0.6547, 0.6718, 0.7291, 1, 1]
        acc = [round(item, 2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        ax = fig.add_subplot(212)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        ax.set_xlabel("λ (COMPAS, Neural Network)", size=14)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)

        ax.set_ylabel("                                              Fairness", size=12)
        plt.tick_params(axis='y', labelsize=10, rotation=90)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="green", label="FedRényi Accuracy(for \n Uniform Over Client)",
                        linestyle="dotted", linewidth=2)
        ax1.set_ylabel("                                                     Accuracy (Data Partition: Dirichlet)",
                       size=12)
        # plt.yticks([0.69, 0.70, 0.71], rotation=90)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        plt.legend(bbox_to_anchor=(1.11, 2.34), borderaxespad=0, fontsize=10, frameon=False)
        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")

        if show_figure:
            plt.show()

    def NN_Data_Partition_Uniform(path, show_figure=False):
        λ_list = ["0.125", "0.25", "0.5", "1", "2", "4"]
        acc = [0.6904, 0.6695, 0.5921, 0.5167, 0.5167, 0.5167]
        FR = [0.6897, 0.5923, 0.6872, 1, 1, 1]
        acc = [round(item, 2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        fig = plt.figure(figsize=(6, 6), dpi=100)
        plt.subplots_adjust(hspace=0.16)

        ax = fig.add_subplot(211)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=10)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        # ax.set_ylabel("Fairness", size=12)
        plt.legend(bbox_to_anchor=(0.15, 1.14), borderaxespad=0, fontsize=10, frameon=False)
        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="blue", label="FedRényi Accuracy (for \n Uniform Over Distribution)",
                        linestyle="dashed", linewidth=2)
        plt.yticks([0.50, 0.55, 0.60, 0.65, 0.70], rotation=90)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        plt.legend(bbox_to_anchor=(0.66, 1.18), borderaxespad=0, fontsize=10, frameon=False)

        acc = [0.4833, 0.5167, 0.6172, 0.4833, 0.4833, 0.477]
        FR = [1, 0.941, 0.9282, 1, 1, 0.9923]
        acc = [round(item, 2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        ax = fig.add_subplot(212)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        ax.set_xlabel("λ (COMPAS, Neural Network)", size=14)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)

        ax.set_ylabel("                                              Fairness", size=12)
        plt.tick_params(axis='y', labelsize=10, rotation=90)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="green", label="FedRényi Accuracy(for \n Uniform Over Client)",
                        linestyle="dotted", linewidth=2)
        ax1.set_ylabel("                                                     Accuracy (Data Partition: Uniform)",
                       size=12)
        plt.yticks([0.50, 0.55, 0.60, 0.65], rotation=90)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        plt.legend(bbox_to_anchor=(1.11, 2.34), borderaxespad=0, fontsize=10, frameon=False)
        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")

        if show_figure:
            plt.show()


    path = "../save_path/figure/COMPAS/LR/Lamda_Data_Partition_Dirichlet.png"
    LR_Data_Partition_Dirichlet(path, show_figure=False)
    path = "../save_path/figure/COMPAS/LR/Lamda_Data_Partition_Uniform.png"
    LR_Data_Partition_Uniform(path, show_figure=False)
    path = "../save_path/figure/COMPAS/NN/Lamda_Data_Partition_Dirichlet.png"
    NN_Data_Partition_Dirichlet(path, show_figure=False)
    path = "../save_path/figure/COMPAS/NN/Lamda_Data_Partition_Uniform.png"
    NN_Data_Partition_Uniform(path, show_figure=False)

def ADULT():
    def LR_Data_Partition_Dirichlet(path, show_figure=False):
        λ_list = ["0.125", "0.25", "0.5", "1", "2", "4"]
        FR = [0.8636, 0.8865, 0.8845, 0.9764, 0.9254, 0.9946]
        acc = [0.857, 0.8568, 0.8545, 0.8469, 0.8483, 0.7652]
        FR = [round(item, 2) for item in FR]
        acc = [round(item,2) for item in acc]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        fig = plt.figure(figsize=(6, 6), dpi=100)
        plt.subplots_adjust(hspace=0.16)

        ax = fig.add_subplot(211)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=10)
        plt.yticks([0.85, 0.90, 0.95, 1.00], rotation=90)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        # ax.set_ylabel("Fairness", size=12)
        plt.legend(bbox_to_anchor=(0.15, 1.14), borderaxespad=0, fontsize=10, frameon=False)
        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="blue", label="FedRényi Accuracy (for \n Uniform Over Distribution)", linestyle="dashed", linewidth=2)
        plt.yticks([0.75, 0.80, 0.85, 0.90], rotation=90)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        plt.legend(bbox_to_anchor=(0.66, 1.18), borderaxespad=0, fontsize=10, frameon=False)

        FR = [0.8636, 0.8865, 0.8882, 0.9764, 0.9254, 0.9946]
        acc = [0.8573, 0.8568, 0.8493, 0.8469, 0.8483, 0.7652]
        FR = [round(item, 2) for item in FR]
        acc = [round(item, 2) for item in acc]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        ax = fig.add_subplot(212)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        ax.set_xlabel("λ (ADULT, Logistic Regression)", size=14)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)

        ax.set_ylabel("                                              Fairness", size=12)
        plt.yticks([0.85, 0.90, 0.95, 1.00], rotation=90)
        plt.tick_params(axis='y', labelsize=10, rotation=90)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="green", label="FedRényi Accuracy (for \n Uniform Over Client)", linestyle="dotted", linewidth=2)
        ax1.set_ylabel("                                                     Accuracy (Data Partition: Dirichlet)", size=12)
        plt.yticks([0.75, 0.80, 0.85, 0.90], rotation=90)
        plt.tick_params(axis='y', labelsize=10)
        plt.legend(bbox_to_anchor=(1.11, 2.34), borderaxespad=0, fontsize=10, frameon=False)
        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")

        if show_figure:
            plt.show()

    def LR_Data_Partition_Uniform(path, show_figure=False):
        λ_list = ["0.125", "0.25", "0.5", "1", "2", "4"]
        acc = [0.8321, 0.8326, 0.8474, 0.8065, 0.7829, 0.7638]
        FR = [0.6606, 0.666, 0.8269, 0.8543, 0.9069, 1]
        acc = [round(item, 2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        fig = plt.figure(figsize=(6, 6), dpi=100)
        plt.subplots_adjust(hspace=0.16)

        ax = fig.add_subplot(211)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=10)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        # ax.set_ylabel("Fairness", size=12)
        plt.legend(bbox_to_anchor=(0.15, 1.14), borderaxespad=0, fontsize=10, frameon=False)
        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="blue", label="FedRényi Accuracy (for \n Uniform Over Distribution)",
                        linestyle="dashed", linewidth=2)
        plt.yticks([0.73, 0.77, 0.81, 0.85], rotation=90)
        plt.tick_params(axis='y', labelsize=10)
        plt.legend(bbox_to_anchor=(0.66, 1.18), borderaxespad=0, fontsize=10, frameon=False)

        acc = [0.8321, 0.8324, 0.7982, 0.7786, 0.7638, 0.7638]
        FR = [0.6568, 0.6658, 0.9117, 0.9285, 1, 1]
        acc = [round(item, 2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        ax = fig.add_subplot(212)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        ax.set_xlabel("λ (ADULT, Logistic Regression)", size=14)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)

        ax.set_ylabel("                                              Fairness", size=12)
        plt.tick_params(axis='y', labelsize=10, rotation=90)

        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="green", label="FedRényi Accuracy (for \n Uniform Over Client)",
                        linestyle="dotted", linewidth=2)
        ax1.set_ylabel("                                                     Accuracy (Data Partition: Uniform)",
                       size=12)
        plt.yticks([0.60, 0.70, 0.80, 0.90, 1], rotation=90)
        plt.tick_params(axis='y', labelsize=10)
        plt.legend(bbox_to_anchor=(1.11, 2.34), borderaxespad=0, fontsize=10, frameon=False)
        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")

        if show_figure:
            plt.show()

    def NN_Data_Partition_Dirichlet(path, show_figure=False):
        λ_list = ["0.125", "0.25", "0.5", "1", "2", "4"]
        acc = [0.8529, 0.8643, 0.7638, 0.7638, 0.7638, 0.7638]
        FR = [0.9312, 0.9556, 1, 1, 1, 1]
        acc = [round(item, 2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        fig = plt.figure(figsize=(6, 6), dpi=100)
        plt.subplots_adjust(hspace=0.16)

        ax = fig.add_subplot(211)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=10)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        # ax.set_ylabel("Fairness", size=12)
        plt.legend(bbox_to_anchor=(0.15, 1.14), borderaxespad=0, fontsize=10, frameon=False)
        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="blue", label="FedRényi Accuracy (for \n Uniform Over Distribution)",
                        linestyle="dashed", linewidth=2)
        plt.yticks([0.75, 0.79, 0.83, 0.87], rotation=90)
        plt.tick_params(axis='y', labelsize=10)
        plt.legend(bbox_to_anchor=(0.66, 1.18), borderaxespad=0, fontsize=10, frameon=False)

        acc = [0.8616, 0.8642, 0.8618, 0.8655, 0.7638, 0.7638]
        FR = [0.8628, 0.9044, 0.8915, 0.9303, 1, 1]
        acc = [round(item, 2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        ax = fig.add_subplot(212)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        ax.set_xlabel("λ (ADULT, Neural Network)", size=14)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)

        ax.set_ylabel("                                              Fairness", size=12)
        plt.yticks([0.85, 0.90, 0.95, 1], rotation=90)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="green", label="FedRényi Accuracy(for \n Uniform Over Client)",
                        linestyle="dotted", linewidth=2)
        ax1.set_ylabel("                                                     Accuracy (Data Partition: Dirichlet)",
                       size=12)
        plt.yticks([0.75, 0.80, 0.85, 0.90], rotation=90)
        plt.tick_params(axis='y', labelsize=10)
        plt.legend(bbox_to_anchor=(1.11, 2.34), borderaxespad=0, fontsize=10, frameon=False)
        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")

        if show_figure:
            plt.show()

    def NN_Data_Partition_Uniform(path, show_figure=False):
        λ_list = ["0.125", "0.25", "0.5", "1", "2", "4"]
        acc = [0.8529, 0.8643, 0.7638, 0.7638, 0.7638, 0.7638]
        FR = [0.9312, 0.9556, 1, 1, 1, 1]
        acc = [round(item, 2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        fig = plt.figure(figsize=(6, 6), dpi=100)
        plt.subplots_adjust(hspace=0.16)

        ax = fig.add_subplot(211)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=10)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        # ax.set_ylabel("Fairness", size=12)
        plt.legend(bbox_to_anchor=(0.15, 1.14), borderaxespad=0, fontsize=10, frameon=False)
        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="blue", label="FedRényi Accuracy (for \n Uniform Over Distribution)",
                        linestyle="dashed", linewidth=2)
        plt.yticks([0.75, 0.79, 0.83, 0.87], rotation=90)
        plt.tick_params(axis='y', labelsize=10)
        plt.legend(bbox_to_anchor=(0.66, 1.18), borderaxespad=0, fontsize=10, frameon=False)

        acc = [0.8616, 0.8642, 0.8618, 0.8655, 0.7638, 0.7638]
        FR = [0.8628, 0.9044, 0.8915, 0.9303, 1, 1]
        acc = [round(item, 2) for item in acc]
        FR = [round(item, 2) for item in FR]
        df = pd.DataFrame(data={"FR": FR, "acc": acc}, index=λ_list)

        ax = fig.add_subplot(212)
        lin1 = ax.plot(df.index, df["FR"], label="Fairness", color='#E50000', linewidth=2)
        ax.set_xlabel("λ (ADULT, Neural Network)", size=14)
        plt.xticks(["0.125", "0.25", "0.5", "1", "2", "4"], rotation=0)

        ax.set_ylabel("                                              Fairness", size=12)
        plt.yticks([0.85, 0.90, 0.95, 1], rotation=90)
        plt.tick_params(axis='y', labelsize=10, rotation=90)
        ax1 = ax.twinx()
        lin2 = ax1.plot(df.index, df["acc"], color="green", label="FedRényi Accuracy(for \n Uniform Over Client)",
                        linestyle="dotted", linewidth=2)
        ax1.set_ylabel("                                                     Accuracy (Data Partition: Uniform)",
                       size=12)
        plt.yticks([0.75, 0.80, 0.85, 0.90], rotation=90)
        plt.tick_params(axis='y', labelsize=10)
        plt.legend(bbox_to_anchor=(1.11, 2.34), borderaxespad=0, fontsize=10, frameon=False)
        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")

        if show_figure:
            plt.show()


    path = "../save_path/figure/ADULT/LR/Lamda_Data_Partition_Dirichlet.png"
    LR_Data_Partition_Dirichlet(path, show_figure=False)
    path = "../save_path/figure/ADULT/LR/Lamda_Data_Partition_Uniform.png"
    LR_Data_Partition_Uniform(path, show_figure=False)
    path = "../save_path/figure/ADULT/NN/Lamda_Data_Partition_Dirichlet.png"
    NN_Data_Partition_Dirichlet(path, show_figure=False)
    path = "../save_path/figure/ADULT/NN/Lamda_Data_Partition_Uniform.png"
    NN_Data_Partition_Uniform(path, show_figure=False)


if __name__ == '__main__':
    DRUG()
    COMPAS()
    ADULT()

