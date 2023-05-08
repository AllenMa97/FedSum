import copy
import os
import json
import difflib

def log_to_table(log_path,
                 no_drop=True,
                 split_strategy="Dirichlet"):
    file_list = os.listdir(log_path)
    file_count = len([_ for _ in file_list if "results.json" in _])
    global_acc_list, uniform_client_acc_list, uniform_distribution_acc_list, FR_list, HM_list = [], [], [], [], []

    file_num = []
    for i in range(file_count):
        with open(os.path.join(log_path, str(i+1)+"_Parameter.json"), "r") as f:
            Parameter_dict = json.load(f)

        if no_drop:
            if Parameter_dict["FL_drop_rate"] != 0:
                continue
        if "Dirichlet" in split_strategy:
            if "Uniform" in Parameter_dict["split_strategy"]:
                continue
        else:
            if "Dirichlet" in Parameter_dict["split_strategy"]:
                continue

        with open(os.path.join(log_path, str(i+1)+"_result.json"), "r") as f:
            result_dict = json.load(f)
            global_acc_list.append(result_dict['global_acc'])
            uniform_client_acc_list.append(result_dict['uniform_client_acc'])
            uniform_distribution_acc_list.append(result_dict['uniform_distribution_acc'])
            FR_list.append(result_dict['FR'])
            HM_list.append(result_dict['HM'])

        file_num.append(i+1)

    tmp_list_1, tmp_list_2, tmp_list_3, tmp_list_4, tmp_list_5, tuple_list = [], [], [], [], [], []
    for j in range(len(global_acc_list)):
        global_acc = round(float(global_acc_list[j]), 4)

        uniform_client_acc = round(float(uniform_client_acc_list[j]), 4)

        uniform_distribution_acc = round(float(uniform_distribution_acc_list[j]), 4)

        FR = round(float(FR_list[j]), 4)

        HM = round(float(HM_list[j]), 4)

        tuple_list.append((file_num[j], global_acc, uniform_client_acc, uniform_distribution_acc, FR, HM))

    global_acc_first_list = copy.deepcopy(sorted(tuple_list, key=lambda _: _[1]))
    global_acc_first_list.reverse()

    uniform_client_acc_first_list = copy.deepcopy(sorted(tuple_list, key=lambda _: _[2]))
    uniform_client_acc_first_list.reverse()

    uniform_distribution_acc_first_list = copy.deepcopy(sorted(tuple_list, key=lambda _: _[3]))
    uniform_distribution_acc_first_list.reverse()

    FR_first_list = copy.deepcopy(sorted(tuple_list, key=lambda _: _[4]))
    FR_first_list.reverse()

    HM_first_list = copy.deepcopy(sorted(tuple_list, key=lambda _: _[5]))
    HM_first_list.reverse()

    print("Finish")


def main(dataset_name, hypothesis, split_strategy):

    # log_path = "../log_path/" + dataset_name +"/FederatedRenyi/" + hypothesis
    # log_to_table(log_path, no_drop=False, split_strategy=split_strategy)

    # log_path = "../log_path/" + dataset_name + "/FedRenyi_uniform_client/" + hypothesis
    # log_to_table(log_path, no_drop=False, split_strategy=split_strategy)
    #
    # log_path = "../log_path/" + dataset_name + "/FederatedFair/" + hypothesis
    # log_to_table(log_path, no_drop=False, split_strategy=split_strategy)
    #
    # log_path = "../log_path/" + dataset_name + "/LCO/" + hypothesis
    # log_to_table(log_path, no_drop=False, split_strategy=split_strategy)
    #
    # log_path = "../log_path/" + dataset_name + "/FederatedAverage/" + hypothesis
    # log_to_table(log_path, no_drop=False, split_strategy=split_strategy)

    log_path = "../fine_tune/" + dataset_name + "/FederatedFair/" + hypothesis
    log_to_table(log_path, no_drop=False, split_strategy=split_strategy)

if __name__ == '__main__':
    # for dataset_name in ["DRUG", "COMPAS", "ADULT"]:
    #     for hypothesis in ["LR", "NN"]:
    #         for split_strategy in ["Dirichlet", "Uniform"]:
    for dataset_name in ["ADULT"]:
        for hypothesis in ["NN"]:
            for split_strategy in ["Dirichlet"]:
                main(dataset_name, hypothesis, split_strategy)


