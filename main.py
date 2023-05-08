import os
import json
import argparse

from tool.logger import *
from tool.utils import check_and_make_the_path, str2bool
from experiment import Experiment


def Argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-test_all", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)
    parser.add_argument("-train_from", default='')

    args = parser.parse_args()
    param_dict = vars(args)
    return param_dict

def main(dataset_name, algorithm, hypothesis, classifier_type, device, param_dict):
    # Hyper-params
    with open("./json/COMMON.json", "r") as f:
        temp_dict = json.load(f)
    param_dict.update(**temp_dict)
    if ("," not in dataset_name) or (", " not in dataset_name):
        with open(os.path.join("./json/", dataset_name + ".json"), "r") as f:
            temp_dict = json.load(f)
        param_dict.update(**temp_dict)

    os.environ["CUDA_VISIBLE_DEVICES"] = param_dict['CUDA_VISIBLE_DEVICES']
    import torch
    if "gpu" in device.lower():
        param_dict['device'] = "cuda" if torch.cuda.is_available() else "cpu"  # Get cpu or gpu device for experiment
    else:
        param_dict['device'] = "cpu"

    FL_drop_rate_list = [0, 0.1]
    epoch_T_communication_I_list = [(3,1), (5, 1), (20, 5)]
    split_strategy_list = ["Uniform", "Dirichlet"]

    param_dict['dataset_name'] = dataset_name
    param_dict['algorithm'] = algorithm
    param_dict['hypothesis'] = hypothesis
    param_dict['classifier_type'] = classifier_type

    # Skipping the unnecessary loop
    if "pFedSum".lower() not in algorithm.lower():
        FL_drop_rate_list = [0]

    # Serial number of experiment
    Experiment_NO = 1
    total_Experiment_NO = len(FL_drop_rate_list) * len(epoch_T_communication_I_list) * len(split_strategy_list)

    # Main Loop
    for split_strategy in split_strategy_list:
        for FL_drop_rate in FL_drop_rate_list:
            param_dict['FL_drop_rate'] = FL_drop_rate
            for algorithm_epoch_T, communication_round_I in epoch_T_communication_I_list:
                param_dict['split_strategy'] = split_strategy
                param_dict['algorithm_epoch_T'] = algorithm_epoch_T
                param_dict['communication_round_I'] = communication_round_I
                ################################################################################################
                # Create the log
                log_path = os.path.join("./log_path", param_dict['dataset_name'], param_dict['algorithm'],
                                        param_dict['hypothesis'])
                check_and_make_the_path(log_path)
                log_path = os.path.join(log_path, str(Experiment_NO))
                param_dict['log_path'] = log_path
                file_handler = logging.FileHandler(log_path + ".txt")
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                ################################################################################################
                # Create the model path
                model_path = os.path.join("./save_path", param_dict['dataset_name'], param_dict['algorithm'],
                                          param_dict['hypothesis'], str(Experiment_NO))
                check_and_make_the_path(model_path)
                param_dict['model_path'] = model_path
                for k in range(param_dict["num_clients_K"]):
                    _ = os.path.join(model_path, "client_" + str(k + 1))
                    check_and_make_the_path(_)
                logger.info(f"Experiment {Experiment_NO}/{total_Experiment_NO} setup finish")
                param_dict['Experiment_NO'] = str(Experiment_NO)
                ################################################################################################
                # Parameter announcement
                logger.info("Parameter announcement")
                for para_key in list(param_dict.keys()):
                    if "_common" in para_key:
                        continue
                    logger.info(f"****** {para_key} : {param_dict[para_key]} ******")
                logger.info("-----------------------------------------------------------------------------")
                ################################################################################################
                # Experiment
                Experiment(param_dict)
                Experiment_NO += 1
                logger.removeHandler(file_handler)
                logger.info("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
                logger.info("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")


if __name__ == '__main__':
    param_dict = Argparse()
    main(dataset_name="CNNDM, GovernmentReport",
         algorithm="FederatedAverage",
         hypothesis="BERTSUMEXT",
         classifier_type="Transformer",
         device="gpu",
         param_dict=param_dict)

