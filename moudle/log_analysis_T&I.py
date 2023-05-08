import copy
import os
import json
import difflib
import numpy as np

def log_to_table(log_path, split_strategy, algorithm_epoch_T):
    file_list = os.listdir(log_path)
    file_count = len([_ for _ in file_list if "results.json" in _])
    global_acc_list, uniform_client_acc_list, uniform_distribution_acc_list, FR_list, HM_list = [], [], [], [], []

    file_num = []
    for i in range(file_count):
        with open(os.path.join(log_path, str(i+1)+"_Parameter.json"), "r") as f:
            Parameter_dict = json.load(f)

        if float(Parameter_dict["algorithm_epoch_T"]) != algorithm_epoch_T:
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

        tuple_list.append((00, file_num[j], global_acc, uniform_client_acc, uniform_distribution_acc, FR, HM))

    global_acc_first_list = copy.deepcopy(sorted(tuple_list, key=lambda _: _[2]))
    global_acc_first_list.reverse()

    uniform_client_acc_first_list = copy.deepcopy(sorted(tuple_list, key=lambda _: _[3]))
    uniform_client_acc_first_list.reverse()

    uniform_distribution_acc_first_list = copy.deepcopy(sorted(tuple_list, key=lambda _: _[4]))
    uniform_distribution_acc_first_list.reverse()

    FR_first_list = copy.deepcopy(sorted(tuple_list, key=lambda _: _[5]))
    FR_first_list.reverse()

    HM_first_list = copy.deepcopy(sorted(tuple_list, key=lambda _: _[6]))
    HM_first_list.reverse()

    try:
        return HM_first_list
    except:
        return 0

def main(dataset_name, hypothesis, split_strategy, algorithm_epoch_T):
    # print("dataset_name:", dataset_name, end="; ")
    print("hypothesis:", hypothesis, end="; ")
    print("split_strategy:", split_strategy, end="; ")
    print("algorithm_epoch_T:", algorithm_epoch_T)


    # log_path = "../log_path/" + dataset_name + "/FedRenyi_uniform_client/" + hypothesis
    # HM = log_to_table(log_path, split_strategy, algorithm_epoch_T)
    #
    # log_path = "../log_path/" + dataset_name +"/FederatedRenyi/" + hypothesis
    # HM = log_to_table(log_path, split_strategy, algorithm_epoch_T)

    log_path = "../fine_tune/" + dataset_name +"/FederatedFair/" + hypothesis
    HM = log_to_table(log_path, split_strategy, algorithm_epoch_T)
    log_path = "../log_path/" + dataset_name + "/FederatedFair/" + hypothesis
    HM_2 = log_to_table(log_path, split_strategy, algorithm_epoch_T)

    acc_list, fr_list, hm_list = [], [], []
    for item in (HM+HM_2):
        acc_list.append(item[2])
        fr_list.append(item[-2])
        hm_list.append(item[-1])

    acc_list = np.array(acc_list)
    fr_list = np.array(fr_list)
    hm_list = np.array(hm_list)

    print("acc:", str(round(acc_list.mean(),2))+"±"+str(round(acc_list.std(),2)))
    print("fr:", str(round(fr_list.mean(),2))+"±"+str(round(fr_list.std(),2)))
    print("hm:", str(round(hm_list.mean(),2))+"±"+str(round(hm_list.std(),2)))

    print(1)

if __name__ == '__main__':
    # for dataset_name in ["DRUG", "COMPAS", "ADULT"]:
    #     for hypothesis in ["LR", "NN"]:
    #         for split_strategy in ["Dirichlet", "Uniform"]:
    for dataset_name in ["DRUG"]:
        for hypothesis in ["LR"]:
            for split_strategy in ["Uniform"]:
                for algorithm_epoch_T in [10, 20, 30, 100, 120, 150]:
                    main(dataset_name, hypothesis, split_strategy, algorithm_epoch_T)
                print("______________________")

