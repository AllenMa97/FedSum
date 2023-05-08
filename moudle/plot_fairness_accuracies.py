import matplotlib.pyplot as plt
import numpy as np

def DRUG():
    def LR_Uniform_Over_Client():
        # # DRUG LR uniform_client#
        FedAvg = ([0.7887], [0.9159])
        FedFair = ([0.7676, 0.7148, 0.5035], [1, 0.9747, 0.9793])
        LCO = ([0.758], [0.96])
        uniform_client = (
            [0.8169, 0.7711, 0.7958, 0.8063, 0.8028],
            [0.931, 1, 0.9977, 0.8739, 0.9553]
        )

        uniform_distribution = (
            [0.8099, 0.7711, 0.7887, 0.7958, 0.8063],
            [0.9606, 1, 0.9974, 0.9928, 0.9974]
        )
        my_axis = [0.765, 0.855, 0.9, 1.005]

        my_xticks = [0.70, 0.75, 0.8, 0.85]
        my_yticks = [0.85, 0.9, 0.95, 1.0]

        path = "../save_path/figure/DRUG/LR/uniform_client.png"
        plt.figure(figsize=(6, 6), dpi=100)

        plt.axis(my_axis)
        plt.xticks(my_xticks)
        plt.yticks(my_yticks)

        FedAvg_acc = FedAvg[0]
        FedAvg_FR = FedAvg[1]

        FedFair_acc = FedFair[0]
        FedFair_FR = FedFair[1]

        LCO_acc = LCO[0]
        LCO_FR = LCO[1]

        FedRenyi_acc = uniform_client[0]
        FedRenyi_FR = uniform_client[1]

        FedRenyi_acc_2 = uniform_distribution[0]
        FedRenyi_FR_2 = uniform_distribution[1]

        FedAverage = plt.scatter(FedAvg_acc, FedAvg_FR, c="black", marker="1", label="FedAverage")
        FedFair = plt.scatter(FedFair_acc, FedFair_FR, c="hotpink", marker="*", label="FedFair")
        LCO = plt.scatter(LCO_acc, LCO_FR, c="blue", marker="o", label="LCO")
        FedRenyi = plt.scatter(FedRenyi_acc, FedRenyi_FR, c="#88c999", edgecolors="y", marker="^",
                               label="FedRenyi (Uniform Over Client)")
        FedRenyi_2 = plt.scatter(FedRenyi_acc_2, FedRenyi_FR_2, c="red", marker="+",
                                 label="FedRenyi (Uniform Over Distribution)")

        plt.title("DRUG LR (Data Partition: Dirichlet)", size=14)

        plt.xlabel("Testing Accuracy (Uniform Over Client)", size=14)
        plt.ylabel("Testing Fairness", size=14)
        plt.legend((FedAverage, FedFair, LCO, FedRenyi, FedRenyi_2),
                   ('FedAverage', 'FedFair', 'LCO', "FedRenyi\n(Uniform Over Client)",
                    "FedRenyi\n(Uniform Over Distribution)"),
                   loc=3)

        # plt.savefig(path)
        # plt.savefig(path[:-3] + "eps")
        # plt.savefig(path[:-3] + "pdf")
        plt.show()
        # plt.close()

    def LR_Uniform_Over_Distribution():
        # # DRUG LR uniform_distribution#
        FedAvg = ([0.7887], [0.9455])
        FedFair = ([0.7676, 0.6866, 0.7817, 0.7465, 0.7676], [1, 0.9251, 0.9773, 0.844, 0.8292])
        LCO = ([0.778], [0.97])
        uniform_client = (
            [0.7958, 0.7711, 0.7782, 0.7887, 0.7817],
            [0.9258, 0.9951, 1, 0.9701, 0.9901]
        )

        uniform_distribution = (
            [0.7993, 0.7782, 0.7852, 0.7787, 0.7852],
            [0.9209, 1, 0.9951, 0.9701, 0.9553]
        )

        my_axis = [0.765, 0.855, 0.9, 1.005]

        my_xticks = [0.70, 0.75, 0.8, 0.85]
        my_yticks = [0.85, 0.9, 0.95, 1.0]

        path = "../save_path/figure/DRUG/LR/uniform_distribution.png"
        plt.figure(figsize=(5, 5), dpi=100)

        plt.axis(my_axis)
        plt.xticks(my_xticks)
        plt.yticks(my_yticks)

        FedAvg_acc = FedAvg[0]
        FedAvg_FR = FedAvg[1]

        FedFair_acc = FedFair[0]
        FedFair_FR = FedFair[1]

        LCO_acc = LCO[0]
        LCO_FR = LCO[1]

        FedRenyi_acc = uniform_client[0]
        FedRenyi_FR = uniform_client[1]

        FedRenyi_acc_2 = uniform_distribution[0]
        FedRenyi_FR_2 = uniform_distribution[1]

        FedAverage = plt.scatter(FedAvg_acc, FedAvg_FR, c="black", marker="1", label="FedAverage")
        FedFair = plt.scatter(FedFair_acc, FedFair_FR, c="hotpink", marker="*", label="FedFair")
        LCO = plt.scatter(LCO_acc, LCO_FR, c="blue", marker="o", label="LCO")
        FedRenyi = plt.scatter(FedRenyi_acc, FedRenyi_FR, c="#88c999", edgecolors="y", marker="^",
                               label="FedRenyi (Uniform Over Client)")
        FedRenyi_2 = plt.scatter(FedRenyi_acc_2, FedRenyi_FR_2, c="red", marker="+",
                                 label="FedRenyi (Uniform Over Distribution)")

        plt.title("DRUG LR (Data Partition: Uniform)")

        plt.xlabel("Testing Accuracy (Uniform Over Distribution)", size=12)
        plt.ylabel("Testing Fairness", size=12)
        plt.legend((FedAverage, FedFair, LCO, FedRenyi, FedRenyi_2),
                   ('FedAverage', 'FedFair', 'LCO', "FedRenyi\n(Uniform Over Client)",
                    "FedRenyi\n(Uniform Over Distribution)"),
                   loc=3)

        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        # plt.show()
        # plt.close()

    def NN_Uniform_Over_Client():
        # DRUG NN uniform_client#
        FedAvg = ([0.75], [0.842])
        FedFair = ([0.7885], [1])
        LCO = ([0.2324], [1])

        uniform_distribution = (
            [0.8063, 0.7887, 0.7676],
            [0.9875, 0.9977, 1]
        )

        uniform_client = (
            [0.8204, 0.7746, 0.7887, 0.8063, 0.8099],
            [0.9974, 1, 0.9977, 0.9974, 0.9826]
        )

        my_axis = [0.205, 0.855, 0.8, 1.005]

        my_xticks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.70, 0.8, 0.9]
        my_yticks = [0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

        path = "../save_path/figure/DRUG/NN/uniform_client.png"
        plt.figure(figsize=(5, 5), dpi=100)

        plt.axis(my_axis)
        plt.xticks(my_xticks)
        plt.yticks(my_yticks)

        FedAvg_acc = FedAvg[0]
        FedAvg_FR = FedAvg[1]

        FedFair_acc = FedFair[0]
        FedFair_FR = FedFair[1]

        LCO_acc = LCO[0]
        LCO_FR = LCO[1]

        FedRenyi_acc = uniform_client[0]
        FedRenyi_FR = uniform_client[1]

        FedRenyi_acc_2 = uniform_distribution[0]
        FedRenyi_FR_2 = uniform_distribution[1]

        FedAverage = plt.scatter(FedAvg_acc, FedAvg_FR, c="black", marker="1", label="FedAverage")
        FedFair = plt.scatter(FedFair_acc, FedFair_FR, c="hotpink", marker="*", label="FedFair")
        LCO = plt.scatter(LCO_acc, LCO_FR, c="blue", marker="o", label="LCO")
        FedRenyi = plt.scatter(FedRenyi_acc, FedRenyi_FR, c="#88c999", edgecolors="y", marker="^",
                               label="FedRenyi (Uniform Over Client)")
        FedRenyi_2 = plt.scatter(FedRenyi_acc_2, FedRenyi_FR_2, c="red", marker="+",
                                 label="FedRenyi (Uniform Over Distribution)")

        plt.title("DRUG NN (Data Partition: Dirichlet)")

        plt.xlabel("Testing Accuracy (Uniform Over Client)", size=12)
        plt.ylabel("Testing Fairness", size=12)
        plt.legend((FedAverage, FedFair, LCO, FedRenyi, FedRenyi_2),
                   ('FedAverage', 'FedFair', 'LCO', "FedRenyi\n(Uniform Over Client)",
                    "FedRenyi\n(Uniform Over Distribution)"),
                   loc=3)

        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        # plt.show()
        # plt.close()

    def NN_Uniform_Over_Distribution():
        # DRUG NN uniform_distribution#
        FedAvg = ([0.7576], [1])
        FedFair = ([0.7885], [1])
        LCO = ([0.2324], [1])
        uniform_distribution = (
            [0.771, 0.7676, 0.784],
            [0.9951, 1, 0.98]
        )

        uniform_client = (
            [0.7993, 0.7782, 0.7676],
            [0.9356, 0.9951, 1]
        )
        my_axis = [0.205, 0.855, 0.95, 1.005]

        my_xticks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.70, 0.8, 0.9]
        my_yticks = [0.95, 1.0]

        path = "../save_path/figure/DRUG/NN/uniform_distribution.png"
        plt.figure(figsize=(5, 5), dpi=100)

        plt.axis(my_axis)
        plt.xticks(my_xticks)
        plt.yticks(my_yticks)

        FedAvg_acc = FedAvg[0]
        FedAvg_FR = FedAvg[1]

        FedFair_acc = FedFair[0]
        FedFair_FR = FedFair[1]

        LCO_acc = LCO[0]
        LCO_FR = LCO[1]

        FedRenyi_acc = uniform_client[0]
        FedRenyi_FR = uniform_client[1]

        FedRenyi_acc_2 = uniform_distribution[0]
        FedRenyi_FR_2 = uniform_distribution[1]

        FedAverage = plt.scatter(FedAvg_acc, FedAvg_FR, c="black", marker="1", label="FedAverage")
        FedFair = plt.scatter(FedFair_acc, FedFair_FR, c="hotpink", marker="*", label="FedFair")
        LCO = plt.scatter(LCO_acc, LCO_FR, c="blue", marker="o", label="LCO")
        FedRenyi = plt.scatter(FedRenyi_acc, FedRenyi_FR, c="#88c999", edgecolors="y", marker="^",
                               label="FedRenyi (Uniform Over Client)")
        FedRenyi_2 = plt.scatter(FedRenyi_acc_2, FedRenyi_FR_2, c="red", marker="+",
                                 label="FedRenyi (Uniform Over Distribution)")

        plt.title("DRUG NN (Data Partition: Uniform)")

        plt.xlabel("Testing Accuracy (Uniform Over Distribution)", size=12)
        plt.ylabel("Testing Fairness", size=12)
        plt.legend((FedAverage, FedFair, LCO, FedRenyi, FedRenyi_2),
                   ('FedAverage', 'FedFair', 'LCO', "FedRenyi\n(Uniform Over Client)",
                    "FedRenyi\n(Uniform Over Distribution)"),
                   loc=3)

        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        # plt.show()
        # plt.close()

    LR_Uniform_Over_Client()
    LR_Uniform_Over_Distribution()

    NN_Uniform_Over_Client()
    NN_Uniform_Over_Distribution()

def COMPAS():
    def LR_Uniform_Over_Client():
        # COMPAS LR uniform_client#
        FedAvg = ([0.7092], [0.6342])
        FedFair = ([0.705, 0.4833, 0.6464], [0.635, 1, 0.8402])
        LCO = ([0.6464], [0.7402])
        uniform_client = (
            [0.7155, 0.6967, 0.7071, 0.7092, 0.7113],
            [0.694, 0.7786, 0.7137, 0.6496, 0.6556]
        )

        uniform_distribution = (
            [0.7176, 0.6946, 0.7071, 0.7113],
            [0.6709, 0.8453, 0.7154, 0.6744]
        )

        my_axis = [0.605, 0.755, 0.605, 0.755]

        my_xticks = [0.55, 0.6, 0.65, 0.7, 0.75]
        my_yticks = [0.55, 0.6, 0.65, 0.7, 0.75]

        path = "../save_path/figure/COMPAS/LR/uniform_client.png"
        plt.figure(figsize=(5, 5), dpi=100)

        plt.axis(my_axis)
        plt.xticks(my_xticks)
        plt.yticks(my_yticks)

        FedAvg_acc = FedAvg[0]
        FedAvg_FR = FedAvg[1]

        FedFair_acc = FedFair[0]
        FedFair_FR = FedFair[1]

        LCO_acc = LCO[0]
        LCO_FR = LCO[1]

        FedRenyi_acc = uniform_client[0]
        FedRenyi_FR = uniform_client[1]

        FedRenyi_acc_2 = uniform_distribution[0]
        FedRenyi_FR_2 = uniform_distribution[1]

        FedAverage = plt.scatter(FedAvg_acc, FedAvg_FR, c="black", marker="1", label="FedAverage")
        FedFair = plt.scatter(FedFair_acc, FedFair_FR, c="hotpink", marker="*", label="FedFair")
        LCO = plt.scatter(LCO_acc, LCO_FR, c="blue", marker="o", label="LCO")
        FedRenyi = plt.scatter(FedRenyi_acc, FedRenyi_FR, c="#88c999", edgecolors="y", marker="^",
                               label="FedRenyi (Uniform Over Client)")
        FedRenyi_2 = plt.scatter(FedRenyi_acc_2, FedRenyi_FR_2, c="red", marker="+",
                                 label="FedRenyi (Uniform Over Distribution)")

        plt.title("COMPAS LR (Data Partition: Dirichlet)")

        plt.xlabel("Testing Accuracy (Uniform Over Client)", size=12)
        plt.ylabel("Testing Fairness", size=12)
        plt.legend((FedAverage, FedFair, LCO, FedRenyi, FedRenyi_2),
                   ('FedAverage', 'FedFair', 'LCO', "FedRenyi\n(Uniform Over Client)",
                    "FedRenyi\n(Uniform Over Distribution)"),
                   loc=3)

        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        # plt.show()
        # plt.close()

    def LR_Uniform_Over_Distribution():
        # COMPAS LR uniform_distribution#
        FedAvg = ([0.7071], [0.6308])
        FedFair = ([0.6548, 0.6381, 0.6987], [0.7786, 0.9701, 0.6043])
        LCO = ([0.6987], [0.6043])
        uniform_client = (
            [0.7113, 0.6527, 0.6695, 0.6799, 0.6925, 0.7029],
            [0.647, 0.7932, 0.7786, 0.7658, 0.7504, 0.6786]
        )

        uniform_distribution = (
            [0.7113, 0.6464, 0.6778, 0.6946, 0.7029],
            [0.6256, 0.7932, 0.7735, 0.6974, 0.6786]
        )

        my_axis = [0.605, 0.725, 0.6, 0.805]

        my_xticks = [0.60, 0.65, 0.7, 0.75]
        my_yticks = [0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8]

        path = "../save_path/figure/COMPAS/LR/uniform_distribution.png"
        plt.figure(figsize=(5, 5), dpi=100)

        plt.axis(my_axis)
        plt.xticks(my_xticks)
        plt.yticks(my_yticks)

        FedAvg_acc = FedAvg[0]
        FedAvg_FR = FedAvg[1]

        FedFair_acc = FedFair[0]
        FedFair_FR = FedFair[1]

        LCO_acc = LCO[0]
        LCO_FR = LCO[1]

        FedRenyi_acc = uniform_client[0]
        FedRenyi_FR = uniform_client[1]

        FedRenyi_acc_2 = uniform_distribution[0]
        FedRenyi_FR_2 = uniform_distribution[1]

        FedAverage = plt.scatter(FedAvg_acc, FedAvg_FR, c="black", marker="1", label="FedAverage")
        FedFair = plt.scatter(FedFair_acc, FedFair_FR, c="hotpink", marker="*", label="FedFair")
        LCO = plt.scatter(LCO_acc, LCO_FR, c="blue", marker="o", label="LCO")
        FedRenyi = plt.scatter(FedRenyi_acc, FedRenyi_FR, c="#88c999", edgecolors="y", marker="^",
                               label="FedRenyi (Uniform Over Client)")
        FedRenyi_2 = plt.scatter(FedRenyi_acc_2, FedRenyi_FR_2, c="red", marker="+",
                                 label="FedRenyi (Uniform Over Distribution)")

        plt.title("COMPAS LR (Data Partition: Uniform)")

        plt.xlabel("Testing Accuracy (Uniform Over Distribution)", size=12)
        plt.ylabel("Testing Fairness", size=12)
        plt.legend((FedAverage, FedFair, LCO, FedRenyi, FedRenyi_2),
                   ('FedAverage', 'FedFair', 'LCO', "FedRenyi\n(Uniform Over Client)",
                    "FedRenyi\n(Uniform Over Distribution)"),
                   loc=3)

        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        # plt.show()
        # plt.close()

    def NN_Uniform_Over_Client():
        # COMPAS NN uniform_client#
        FedAvg = ([0.6757], [0.644])
        FedFair = ([0.6841, 0.6234, 0.5167], [0.6564, 0.7915, 1])
        LCO = ([0.4833], [1])
        uniform_distribution = (
            [0.7176, 0.5188, 0.5293, 0.6444, 0.6904],
            [0.6179, 1, 0.9923, 0.8239, 0.8085]
        )
        uniform_client = (
            [0.7176, 0.5439, 0.6213, 0.659, 0.6862],
            [0.7051, 0.9966, 0.9778, 0.8897, 0.7974]
        )

        my_axis = [0.455, 0.755, 0.605, 1.005]

        my_xticks = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        my_yticks = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

        path = "../save_path/figure/COMPAS/NN/uniform_client.png"
        plt.figure(figsize=(5, 5), dpi=100)

        plt.axis(my_axis)
        plt.xticks(my_xticks)
        plt.yticks(my_yticks)

        FedAvg_acc = FedAvg[0]
        FedAvg_FR = FedAvg[1]

        FedFair_acc = FedFair[0]
        FedFair_FR = FedFair[1]

        LCO_acc = LCO[0]
        LCO_FR = LCO[1]

        FedRenyi_acc = uniform_client[0]
        FedRenyi_FR = uniform_client[1]

        FedRenyi_acc_2 = uniform_distribution[0]
        FedRenyi_FR_2 = uniform_distribution[1]

        FedAverage = plt.scatter(FedAvg_acc, FedAvg_FR, c="black", marker="1", label="FedAverage")
        FedFair = plt.scatter(FedFair_acc, FedFair_FR, c="hotpink", marker="*", label="FedFair")
        LCO = plt.scatter(LCO_acc, LCO_FR, c="blue", marker="o", label="LCO")
        FedRenyi = plt.scatter(FedRenyi_acc, FedRenyi_FR, c="#88c999", edgecolors="y", marker="^",
                               label="FedRenyi (Uniform Over Client)")
        FedRenyi_2 = plt.scatter(FedRenyi_acc_2, FedRenyi_FR_2, c="red", marker="+",
                                 label="FedRenyi (Uniform Over Distribution)")

        plt.title("COMPAS NN (Data Partition: Dirichlet)")

        plt.xlabel("Testing Accuracy (Uniform Over Client)", size=12)
        plt.ylabel("Testing Fairness", size=12)
        plt.legend((FedAverage, FedFair, LCO, FedRenyi, FedRenyi_2),
                   ('FedAverage', 'FedFair', 'LCO', "FedRenyi\n(Uniform Over Client)", "FedRenyi\n(Uniform Over Distribution)"),
                   loc=3)

        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        # plt.show()
        # plt.close()

    def NN_Uniform_Over_Distribution():
        # COMPAS NN uniform_distribution#
        FedAvg = ([0.6234], [0.7])
        FedFair = ([0.6967, 0.4833, 0.5272], [0.6137, 1, 0.8513])
        LCO = ([0.4833], [1])
        uniform_distribution = (
            [0.6987, 0.5293, 0.6276, 0.6862, 0.6904],
            [0.6462, 1, 0.9162, 0.8496, 0.7179]
        )
        uniform_client = (
            [0.7155, 0.5251, 0.6109, 0.6339, 0.7071, 0.7113],
            [0.6419, 1, 0.9949, 0.8778, 0.8111, 0.665]
        )
        my_axis = [0.455, 0.705, 0.605, 1.005]

        my_xticks = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        my_yticks = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

        path = "../save_path/figure/COMPAS/NN/uniform_distribution.png"
        plt.figure(figsize=(5, 5), dpi=100)

        plt.axis(my_axis)
        plt.xticks(my_xticks)
        plt.yticks(my_yticks)

        FedAvg_acc = FedAvg[0]
        FedAvg_FR = FedAvg[1]

        FedFair_acc = FedFair[0]
        FedFair_FR = FedFair[1]

        LCO_acc = LCO[0]
        LCO_FR = LCO[1]

        FedRenyi_acc = uniform_client[0]
        FedRenyi_FR = uniform_client[1]

        FedRenyi_acc_2 = uniform_distribution[0]
        FedRenyi_FR_2 = uniform_distribution[1]

        FedAverage = plt.scatter(FedAvg_acc, FedAvg_FR, c="black", marker="1", label="FedAverage")
        FedFair = plt.scatter(FedFair_acc, FedFair_FR, c="hotpink", marker="*", label="FedFair")
        LCO = plt.scatter(LCO_acc, LCO_FR, c="blue", marker="o", label="LCO")
        FedRenyi = plt.scatter(FedRenyi_acc, FedRenyi_FR, c="#88c999", edgecolors="y", marker="^",
                               label="FedRenyi (Uniform Over Client)")
        FedRenyi_2 = plt.scatter(FedRenyi_acc_2, FedRenyi_FR_2, c="red", marker="+",
                                 label="FedRenyi (Uniform Over Distribution)")

        plt.title("COMPAS NN (Data Partition: Uniform)")

        plt.xlabel("Testing Accuracy (Uniform Over Distribution)", size=12)
        plt.ylabel("Testing Fairness", size=12)
        plt.legend((FedAverage, FedFair, LCO, FedRenyi, FedRenyi_2),
                   ('FedAverage', 'FedFair', 'LCO', "FedRenyi\n(Uniform Over Client)", "FedRenyi\n(Uniform Over Distribution)"),
                   loc=3)

        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        # plt.show()
        # plt.close()

    LR_Uniform_Over_Client()
    LR_Uniform_Over_Distribution()

    NN_Uniform_Over_Client()
    NN_Uniform_Over_Distribution()

def ADULT():
    def LR_Uniform_Over_Client():
        # ADULT LR uniform_client#
        FedAvg = ([0.8477], [0.8514])
        FedFair = ([0.761], [1])
        LCO = ([0.7274], [0.8116])
        uniform_client = (
            [0.8613, 0.7638, 0.7652, 0.7758, 0.8609],
            [0.8937, 1, 0.994, 0.9444, 0.9369]
        )
        uniform_distribution = (
            [0.866, 0.8456, 0.8415, 0.8469, 0.8549],
            [0.9316, 1, 0.9986, 0.9764, 0.9639]
        )

        my_axis = [0.705, 0.905, 0.655, 1.05]

        my_xticks = [0.7, 0.75, 0.8, 0.85, 0.9]
        my_yticks = [0.75, 0.8, 0.85, 0.9, 0.95, 1]
        path = "../save_path/figure/ADULT/LR/uniform_client.png"
        plt.figure(figsize=(5, 5), dpi=100)

        plt.axis(my_axis)
        plt.xticks(my_xticks)
        plt.yticks(my_yticks)

        FedAvg_acc = FedAvg[0]
        FedAvg_FR = FedAvg[1]

        FedFair_acc = FedFair[0]
        FedFair_FR = FedFair[1]

        LCO_acc = LCO[0]
        LCO_FR = LCO[1]

        FedRenyi_acc = uniform_client[0]
        FedRenyi_FR = uniform_client[1]

        FedRenyi_acc_2 = uniform_distribution[0]
        FedRenyi_FR_2 = uniform_distribution[1]

        FedAverage = plt.scatter(FedAvg_acc, FedAvg_FR, c="black", marker="1", label="FedAverage")
        FedFair = plt.scatter(FedFair_acc, FedFair_FR, c="hotpink", marker="*", label="FedFair")
        LCO = plt.scatter(LCO_acc, LCO_FR, c="blue", marker="o", label="LCO")
        FedRenyi = plt.scatter(FedRenyi_acc, FedRenyi_FR, c="#88c999", edgecolors="y", marker="^",
                               label="FedRenyi (Uniform Over Client)")
        FedRenyi_2 = plt.scatter(FedRenyi_acc_2, FedRenyi_FR_2, c="red", marker="+",
                                 label="FedRenyi (Uniform Over Distribution)")

        plt.title("ADULT LR (Data Partition: Dirichlet)")

        plt.xlabel("Testing Accuracy (Uniform Over Client)", size=12)
        plt.ylabel("Testing Fairness", size=12)
        plt.legend((FedAverage, FedFair, LCO, FedRenyi, FedRenyi_2),
                   ('FedAverage', 'FedFair', 'LCO', "FedRenyi\n(Uniform Over Client)",
                    "FedRenyi\n(Uniform Over Distribution)"),
                   loc=3)

        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        # plt.show()
        # plt.close()

    def LR_Uniform_Over_Distribution():
        # ADULT LR uniform_distribution#
        FedAvg = ([0.8441], [0.7955])
        FedFair = ([0.7936], [0.6319])
        LCO = ([0.7613], [0.6353])
        uniform_client = (
            [0.8488, 0.7638, 0.7773, 0.7954, 0.8032],
            [0.8627, 1, 0.9402, 0.9289, 0.8779]
        )
        uniform_distribution = (
            [0.7638, 0.7756, 0.7952, 0.8033, 0.8474],
            [1, 0.9431, 0.9347, 0.8832, 0.8707]
        )

        my_axis = [0.755, 0.855, 0.55, 1.05]

        my_xticks = [0.7, 0.75, 0.8, 0.85]
        my_yticks = [0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        path = "../save_path/figure/ADULT/LR/uniform_distribution.png"
        plt.figure(figsize=(5, 5), dpi=100)

        plt.axis(my_axis)
        plt.xticks(my_xticks)
        plt.yticks(my_yticks)

        FedAvg_acc = FedAvg[0]
        FedAvg_FR = FedAvg[1]

        FedFair_acc = FedFair[0]
        FedFair_FR = FedFair[1]

        LCO_acc = LCO[0]
        LCO_FR = LCO[1]

        FedRenyi_acc = uniform_client[0]
        FedRenyi_FR = uniform_client[1]

        FedRenyi_acc_2 = uniform_distribution[0]
        FedRenyi_FR_2 = uniform_distribution[1]

        FedAverage = plt.scatter(FedAvg_acc, FedAvg_FR, c="black", marker="1", label="FedAverage")
        FedFair = plt.scatter(FedFair_acc, FedFair_FR, c="hotpink", marker="*", label="FedFair")
        LCO = plt.scatter(LCO_acc, LCO_FR, c="blue", marker="o", label="LCO")
        FedRenyi = plt.scatter(FedRenyi_acc, FedRenyi_FR, c="#88c999", edgecolors="y", marker="^",
                               label="FedRenyi (Uniform Over Client)")
        FedRenyi_2 = plt.scatter(FedRenyi_acc_2, FedRenyi_FR_2, c="red", marker="+",
                                 label="FedRenyi (Uniform Over Distribution)")

        plt.title("ADULT LR (Data Partition: Uniform)")

        plt.xlabel("Testing Accuracy (Uniform Over Distribution)", size=12)
        plt.ylabel("Testing Fairness", size=12)
        plt.legend((FedAverage, FedFair, LCO, FedRenyi, FedRenyi_2),
                   ('FedAverage', 'FedFair', 'LCO', "FedRenyi\n(Uniform Over Client)",
                    "FedRenyi\n(Uniform Over Distribution)"),
                   loc=3)

        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        # plt.show()
        # plt.close()

    def NN_Uniform_Over_Client():
        # ADULT NN uniform_client#
        FedAvg = ([0.812], [0.7828])
        FedFair = ([0.7738], [1])
        LCO = ([0.7538], [0.987])
        uniform_client = (
            [0.8698, 0.7638, 0.8507, 0.8624, 0.8662],
            [0.8772, 1, 0.9963, 0.9867, 0.9616]
        )
        uniform_distribution = (
            [0.888, 0.7638, 0.8467, 0.8524, 0.8662],
            [0.8472, 1, 0.9963, 0.981, 0.9616]
        )
        my_axis = [0.755, 0.905, 0.605, 1.005]

        my_xticks = [0.7, 0.75, 0.8, 0.85, 0.9]
        my_yticks = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        path = "../save_path/figure/ADULT/NN/uniform_client.png"
        plt.figure(figsize=(5, 5), dpi=100)

        plt.axis(my_axis)
        plt.xticks(my_xticks)
        plt.yticks(my_yticks)

        FedAvg_acc = FedAvg[0]
        FedAvg_FR = FedAvg[1]

        FedFair_acc = FedFair[0]
        FedFair_FR = FedFair[1]

        LCO_acc = LCO[0]
        LCO_FR = LCO[1]

        FedRenyi_acc = uniform_client[0]
        FedRenyi_FR = uniform_client[1]

        FedRenyi_acc_2 = uniform_distribution[0]
        FedRenyi_FR_2 = uniform_distribution[1]

        FedAverage = plt.scatter(FedAvg_acc, FedAvg_FR, c="black", marker="1", label="FedAverage")
        FedFair = plt.scatter(FedFair_acc, FedFair_FR, c="hotpink", marker="*", label="FedFair")
        LCO = plt.scatter(LCO_acc, LCO_FR, c="blue", marker="o", label="LCO")
        FedRenyi = plt.scatter(FedRenyi_acc, FedRenyi_FR, c="#88c999", edgecolors="y", marker="^",
                               label="FedRenyi (Uniform Over Client)")
        FedRenyi_2 = plt.scatter(FedRenyi_acc_2, FedRenyi_FR_2, c="red", marker="+",
                                 label="FedRenyi (Uniform Over Distribution)")

        plt.title("ADULT NN (Data Partition: Dirichlet)")

        plt.xlabel("Testing Accuracy (Uniform Over Client)", size=12)
        plt.ylabel("Testing Fairness", size=12)
        plt.legend((FedAverage, FedFair, LCO, FedRenyi, FedRenyi_2),
                   ('FedAverage', 'FedFair', 'LCO', "FedRenyi\n(Uniform Over Client)", "FedRenyi\n(Uniform Over Distribution)"),
                   loc=3)

        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        # plt.show()
        # plt.close()

    def NN_Uniform_Over_Distribution():
        # ADULT NN uniform_distribution#
        FedAvg = ([0.7654], [0.9917])
        FedFair = ([0.7618], [1])
        LCO = ([0.7538], [1])
        uniform_client = (
            [0.8231, 0.7638, 0.7911, 0.8045, 0.8142],
            [0.8581, 1, 0.9422, 0.8776, 0.8781]
        )
        uniform_distribution = (
            [0.8287, 0.7638, 0.7851, 0.8045, 0.8262],
            [0.8561, 1, 0.9455, 0.8776, 0.8721]
        )

        my_axis = [0.755, 0.855, 0.855, 1.005]

        my_xticks = [0.7, 0.75, 0.80, 0.85]
        my_yticks = [0.75, 0.80, 0.85, 0.9, 0.95, 1.0]
        path = "../save_path/figure/ADULT/NN/uniform_distribution.png"
        plt.figure(figsize=(5, 5), dpi=100)

        plt.axis(my_axis)
        plt.xticks(my_xticks)
        plt.yticks(my_yticks)

        FedAvg_acc = FedAvg[0]
        FedAvg_FR = FedAvg[1]

        FedFair_acc = FedFair[0]
        FedFair_FR = FedFair[1]

        LCO_acc = LCO[0]
        LCO_FR = LCO[1]

        FedRenyi_acc = uniform_client[0]
        FedRenyi_FR = uniform_client[1]

        FedRenyi_acc_2 = uniform_distribution[0]
        FedRenyi_FR_2 = uniform_distribution[1]

        FedAverage = plt.scatter(FedAvg_acc, FedAvg_FR, c="black", marker="1", label="FedAverage")
        FedFair = plt.scatter(FedFair_acc, FedFair_FR, c="hotpink", marker="*", label="FedFair")
        LCO = plt.scatter(LCO_acc, LCO_FR, c="blue", marker="o", label="LCO")
        FedRenyi = plt.scatter(FedRenyi_acc, FedRenyi_FR, c="#88c999", edgecolors="y", marker="^",
                               label="FedRenyi (Uniform Over Client)")
        FedRenyi_2 = plt.scatter(FedRenyi_acc_2, FedRenyi_FR_2, c="red", marker="+",
                                 label="FedRenyi (Uniform Over Distribution)")

        plt.title("ADULT NN (Data Partition: Uniform)")

        plt.xlabel("Testing Accuracy (Uniform Over Distribution)", size=12)
        plt.ylabel("Testing Fairness", size=12)
        plt.legend((FedAverage, FedFair, LCO, FedRenyi, FedRenyi_2),
                   ('FedAverage', 'FedFair', 'LCO', "FedRenyi\n(Uniform Over Client)", "FedRenyi\n(Uniform Over Distribution)"),
                   loc=3)

        plt.savefig(path)
        plt.savefig(path[:-3] + "eps")
        plt.savefig(path[:-3] + "pdf")
        # plt.show()
        # plt.close()

    LR_Uniform_Over_Client()
    LR_Uniform_Over_Distribution()

    NN_Uniform_Over_Client()
    NN_Uniform_Over_Distribution()


if __name__ == '__main__':

    DRUG()
    COMPAS()
    ADULT()