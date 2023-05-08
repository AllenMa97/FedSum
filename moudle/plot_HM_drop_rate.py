import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter

def DRUG():
    def DRUG_LR(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False):
        λ_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        fig = plt.figure(figsize=(6, 5), dpi=100)
        ###############################################################
        ax = fig.add_subplot(111)

        uniform_client = [round(i, 2) for i in renyi_uniform_client]
        uniform_distribution = [round(i, 2) for i in renyi_uniform_distribution]

        df = pd.DataFrame(data={"uniform_client": uniform_client, "uniform_distribution": uniform_distribution},
                          index=λ_list)
        lin1 = ax.plot(df.index, df["uniform_client"], label="FedRenyi (Uniform Client)",
                       color='red', linestyle="dashed", linewidth=2)
        plt.legend(fontsize=15, loc=2)

        lin2 = ax.plot(df.index, df["uniform_distribution"],
                       label="FedRenyi (Uniform Distribution)", color="blue",
                       linestyle="dashdot", linewidth=2)
        plt.legend(fontsize=15, loc=2)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.title("DRUG LR (Data Partition: " + str(Data_Partition) + ";   λ = " + str(lamda) + ")")

        if "Dirichlet" in Data_Partition:
            plt.yticks([0.86, 0.87, 0.88, 0.89, 0.90], rotation=0)
        plt.tick_params(axis='y', labelsize=11)
        ax.set_ylabel("Harmonic Mean", size=15)

        plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=11)
        ax.set_xlabel("Drop Rate", size=18)
        plt.savefig(path)

        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        if show_figure:
            plt.show()

    def DRUG_NN(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False):
        λ_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        fig = plt.figure(figsize=(6, 5), dpi=100)
        ###############################################################
        ax = fig.add_subplot(111)

        uniform_client = [round(i, 2) for i in renyi_uniform_client]
        uniform_distribution = [round(i, 2) for i in renyi_uniform_distribution]

        df = pd.DataFrame(data={"uniform_client": uniform_client, "uniform_distribution": uniform_distribution},
                          index=λ_list)
        lin1 = ax.plot(df.index, df["uniform_client"], label="FedRenyi (Uniform Client)",
                       color='red', linestyle="dashed", linewidth=2)
        plt.legend(fontsize=15, loc=2)

        lin2 = ax.plot(df.index, df["uniform_distribution"],
                       label="FedRenyi (Uniform Distribution)", color="blue",
                       linestyle="dashdot", linewidth=2)
        plt.legend(fontsize=15, loc=2)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.title("DRUG NN (Data Partition: " + str(Data_Partition) + ";   λ = " + str(lamda) + ")")

        if "Dirichlet" in Data_Partition:
            plt.yticks([0.86, 0.87, 0.88, 0.89, 0.90], rotation=0)
        plt.tick_params(axis='y', labelsize=11)
        ax.set_ylabel("Harmonic Mean", size=15)

        plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=11)
        ax.set_xlabel("Drop Rate", size=18)
        plt.savefig(path)

        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        if show_figure:
            plt.show()

    lamda = 1
    path = "../save_path/figure/DRUG/LR/Uniform_drop_rate.png"
    Data_Partition = "Dirichlet"
    renyi_uniform_client = [0.8685, 0.8708, 0.8854, 0.8685, 0.8685, 0.8685]
    renyi_uniform_distribution = [0.8708, 0.8685, 0.8711, 0.8685, 0.8685, 0.8877]
    DRUG_LR(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False)

    path = "../save_path/figure/DRUG/LR/Dirichlet_drop_rate.png"
    Data_Partition = "Uniform"
    renyi_uniform_client = [0.873, 0.873, 0.8685, 0.8685, 0.8698, 0.8708]
    renyi_uniform_distribution = [0.873, 0.8711, 0.8685, 0.8685, 0.8717, 0.8708]
    DRUG_LR(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False)

    lamda = 0.125
    path = "../save_path/figure/DRUG/NN/Uniform_drop_rate.png"
    Data_Partition = "Dirichlet"
    renyi_uniform_client = [0.8685, 0.8802, 0.8742, 0.8806, 0.8685, 0.8685]
    renyi_uniform_distribution = [0.8769, 0.8704, 0.8878, 0.8685, 0.8685, 0.8685]
    DRUG_NN(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False)

    path = "../save_path/figure/DRUG/NN/Dirichlet_drop_rate.png"
    Data_Partition = "Uniform"
    renyi_uniform_client = [0.8685, 0.8685, 0.8708, 0.8734, 0.8685, 0.8685]
    renyi_uniform_distribution = [0.8685, 0.8685, 0.8685, 0.8685, 0.8685, 0.8685]
    DRUG_NN(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False)

def COMPAS():

    def COMPAS_LR(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False):
        λ_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        fig = plt.figure(figsize=(6, 5), dpi=100)
        ###############################################################
        ax = fig.add_subplot(111)

        uniform_client = [round(i, 2) for i in renyi_uniform_client]
        uniform_distribution = [round(i, 2) for i in renyi_uniform_distribution]

        df = pd.DataFrame(data={"uniform_client": uniform_client, "uniform_distribution": uniform_distribution},
                          index=λ_list)
        lin1 = ax.plot(df.index, df["uniform_client"], label="FedRenyi (Uniform Client)",
                       color='red', linestyle="dashed", linewidth=2)
        plt.legend(fontsize=15, loc=2)

        lin2 = ax.plot(df.index, df["uniform_distribution"],
                       label="FedRenyi (Uniform Distribution)", color="blue",
                       linestyle="dashdot", linewidth=2)
        plt.legend(fontsize=15, loc=2)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.title("COMPAS LR (Data Partition: " + str(Data_Partition) + ";   λ = " + str(lamda) + ")")

        if "Dirichlet" in Data_Partition:
            plt.yticks([0.69, 0.70, 0.71, 0.72, 0.73], rotation=0)
        else:
            plt.yticks([0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73], rotation=0)

        plt.tick_params(axis='y' , labelsize=11)
        ax.set_ylabel("Harmonic Mean", size=15 )

        plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=11)
        ax.set_xlabel("Drop Rate", size=18)
        plt.savefig(path)

        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        if show_figure:
            plt.show()

    def COMPAS_NN(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False):
        λ_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        fig = plt.figure(figsize=(6, 5), dpi=100)
        ###############################################################
        ax = fig.add_subplot(111)

        uniform_client = [round(i, 2) for i in renyi_uniform_client]
        uniform_distribution = [round(i, 2) for i in renyi_uniform_distribution]

        df = pd.DataFrame(data={"uniform_client": uniform_client, "uniform_distribution": uniform_distribution},
                          index=λ_list)
        lin1 = ax.plot(df.index, df["uniform_client"], label="FedRenyi (Uniform Client)",
                       color='red', linestyle="dashed", linewidth=2)
        plt.legend(fontsize=15, loc=2)

        lin2 = ax.plot(df.index, df["uniform_distribution"],
                       label="FedRenyi (Uniform Distribution)", color="blue",
                       linestyle="dashdot", linewidth=2)
        plt.legend(fontsize=15, loc=2)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.title("COMPAS NN (Data Partition: " + str(Data_Partition) + ";   λ = " + str(lamda) + ")")

        if "Dirichlet" in Data_Partition:
            plt.yticks([0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77], rotation=0)
        else:
            plt.yticks([0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73,  0.74], rotation=0)

        plt.tick_params(axis='y' , labelsize=11)
        ax.set_ylabel("Harmonic Mean", size=15 )

        plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=11)
        ax.set_xlabel("Drop Rate", size=18)
        plt.savefig(path)

        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        if show_figure:
            plt.show()

    lamda = 1
    path = "../save_path/figure/COMPAS/LR/Uniform_drop_rate.png"
    Data_Partition = "Dirichlet"
    renyi_uniform_client = [0.7055, 0.7006, 0.713, 0.7068, 0.7104, 0.7188]
    renyi_uniform_distribution = [0.7082, 0.7124, 0.6971, 0.7233, 0.7192, 0.7233]
    COMPAS_LR(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False)

    path = "../save_path/figure/COMPAS/LR/Dirichlet_drop_rate.png"
    Data_Partition = "Uniform"
    renyi_uniform_client = [0.6777, 0.6933, 0.6937, 0.6955, 0.716, 0.6856]
    renyi_uniform_distribution = [0.6777, 0.6933, 0.6937, 0.7102, 0.7131, 0.6837]
    COMPAS_LR(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False)

    lamda = 0.125
    path = "../save_path/figure/COMPAS/NN/Uniform_drop_rate.png"
    Data_Partition = "Dirichlet"
    renyi_uniform_client = [0.7036, 0.706, 0.712, 0.7165, 0.7598, 0.7423]
    renyi_uniform_distribution = [0.7133, 0.7288, 0.6954, 0.7036, 0.7006, 0.7137]
    COMPAS_NN(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False)

    path = "../save_path/figure/COMPAS/NN/Dirichlet_drop_rate.png"
    Data_Partition = "Uniform"
    renyi_uniform_client = [0.678, 0.688, 0.7032, 0.7049, 0.7272, 0.7223]
    renyi_uniform_distribution = [0.6901, 0.6814, 0.6995, 0.6814, 0.6814, 0.6814]
    COMPAS_NN(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False)

def ADULT():
    def ADULT_LR(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False):
        λ_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        fig = plt.figure(figsize=(6, 5), dpi=100)
        ###############################################################
        ax = fig.add_subplot(111)

        uniform_client = [round(i, 2) for i in renyi_uniform_client]
        uniform_distribution = [round(i, 2) for i in renyi_uniform_distribution]

        df = pd.DataFrame(data={"uniform_client": uniform_client, "uniform_distribution": uniform_distribution},
                          index=λ_list)
        lin1 = ax.plot(df.index, df["uniform_client"], label="FedRenyi (Uniform Client)",
                       color='red', linestyle="dashed", linewidth=2)
        plt.legend(fontsize=15, loc=2)

        lin2 = ax.plot(df.index, df["uniform_distribution"],
                       label="FedRenyi (Uniform Distribution)", color="blue",
                       linestyle="dashdot", linewidth=2)
        plt.legend(fontsize=15, loc=2)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.title("ADULT LR (Data Partition: " + str(Data_Partition) + ";   λ = " + str(lamda) + ")")

        if "Dirichlet" in Data_Partition:
            plt.yticks([0.87, 0.88, 0.89, 0.90, 0.91, 0.92], rotation=0)
        else:
            plt.yticks([0.79, 0.83, 0.87, 0.91], rotation=0)

        plt.tick_params(axis='y' , labelsize=11)
        ax.set_ylabel("Harmonic Mean", size=15 )

        plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=11)
        ax.set_xlabel("Drop Rate", size=18)
        plt.savefig(path)

        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        if show_figure:
            plt.show()

    def ADULT_NN(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False):
        λ_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        fig = plt.figure(figsize=(6, 5), dpi=100)
        ###############################################################
        ax = fig.add_subplot(111)

        uniform_client = [round(i, 2) for i in renyi_uniform_client]
        uniform_distribution = [round(i, 2) for i in renyi_uniform_distribution]

        df = pd.DataFrame(data={"uniform_client": uniform_client, "uniform_distribution": uniform_distribution},
                          index=λ_list)
        lin1 = ax.plot(df.index, df["uniform_client"], label="FedRenyi (Uniform Client)",
                       color='red', linestyle="dashed", linewidth=2)
        plt.legend(fontsize=15, loc=2)

        lin2 = ax.plot(df.index, df["uniform_distribution"],
                       label="FedRenyi (Uniform Distribution)", color="blue",
                       linestyle="dashdot", linewidth=2)
        plt.legend(fontsize=15, loc=2)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.title("ADULT NN (Data Partition: " + str(Data_Partition) + ";   λ = " + str(lamda) + ")")

        if "Dirichlet" in Data_Partition:
            plt.yticks([0.80, 0.85, 0.90, 0.95, 1], rotation=0)
        else:
            plt.yticks([0.80, 0.85, 0.90, 0.95, 1], rotation=0)

        plt.tick_params(axis='y' , labelsize=11)
        ax.set_ylabel("Harmonic Mean", size=15 )

        plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], rotation=0)
        plt.tick_params(axis='x', colors='black', labelsize=11)
        ax.set_xlabel("Drop Rate", size=18)
        plt.savefig(path)

        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        if show_figure:
            plt.show()

    lamda = 1
    path = "../save_path/figure/ADULT/LR/Dirichlet_drop_rate.png"
    Data_Partition = "Dirichlet"
    renyi_uniform_client = [ 0.8661, 0.8667, 0.8737, 0.873, 0.8658, 0.8772]
    renyi_uniform_distribution = [ 0.907, 0.8875, 0.8722, 0.888, 0.8802, 0.8987]
    ADULT_LR(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False)

    path = "../save_path/figure/ADULT/LR/Uniform_drop_rate.png"
    Data_Partition = "Uniform"
    renyi_uniform_client = [ 0.8661, 0.8661, 0.8661, 0.8661, 0.8661, 0.8661]
    renyi_uniform_distribution = [ 0.8661, 0.8661, 0.8661, 0.8661, 0.8661, 0.8661 ]
    ADULT_LR(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False)

    lamda = 0.125
    path = "../save_path/figure/ADULT/NN/Dirichlet_drop_rate.png"
    Data_Partition = "Dirichlet"
    renyi_uniform_client = [ 0.8903, 0.9085, 0.8746, 0.8901, 0.8701, 0.9006]
    renyi_uniform_distribution = [ 0.8903, 0.9085, 0.8746, 0.8901, 0.8701, 0.9006]
    ADULT_NN(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False)

    path = "../save_path/figure/ADULT/NN/Uniform_drop_rate.png"
    Data_Partition = "Uniform"
    renyi_uniform_client = [ 0.8661, 0.8661, 0.8661, 0.8661, 0.8661, 0.8661]
    renyi_uniform_distribution = [ 0.8661, 0.8661, 0.8661, 0.8661, 0.8661, 0.8661]
    ADULT_NN(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False)

if __name__ == '__main__':
    DRUG()
    COMPAS()
    ADULT()

    # lamda = 1
    # path = "../save_path/figure/ADULT/LR/Uniform_drop_rate.png"
    # Data_Partition = "Dirichlet"
    # renyi_uniform_client = [ 0.8661, 0.8667, 0.8737, 0.873, 0.8658, 0.8772]
    # renyi_uniform_distribution = [ 0.907, 0.8875, 0.8722, 0.888, 0.8802, 0.8987]
    # ADULT_LR(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False)
    #
    # path = "../save_path/figure/ADULT/LR/Dirichlet_drop_rate.png"
    # Data_Partition = "Uniform"
    # renyi_uniform_client = [ 0.8661, 0.8661, 0.8661, 0.8661, 0.8661, 0.8661]
    # renyi_uniform_distribution = [ 0.8661, 0.8661, 0.8661, 0.8661, 0.8661, 0.8661 ]
    # ADULT_LR(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False)

    # lamda = 0.125
    # path = "../save_path/figure/ADULT/NN/Uniform_drop_rate.png"
    # Data_Partition = "Dirichlet"
    # renyi_uniform_client = [0.7036, 0.706, 0.712, 0.7165, 0.7598, 0.7423]
    # renyi_uniform_distribution = [0.7133, 0.7288, 0.6954, 0.7036, 0.7006, 0.7137]
    # ADULT_NN(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False)
    #
    # path = "../save_path/figure/ADULT/NN/Dirichlet_drop_rate.png"
    # Data_Partition = "Uniform"
    # renyi_uniform_client = [0.678, 0.688, 0.7032, 0.7049, 0.7272, 0.7223]
    # renyi_uniform_distribution = [0.6901, 0.6814, 0.6995, 0.6814, 0.6814, 0.6814]
    # ADULT_NN(path, Data_Partition, renyi_uniform_client, renyi_uniform_distribution, lamda, show_figure=False)
