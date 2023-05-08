import os
import pickle
import torch
from hypothesis.BERTSUMEXT import ExtSummarizer
from moudle.dataset import get_CNNDM_dataset, get_WikiHow_dataset, get_GovernmentReport_dataset, get_PubMed_dataset
from moudle.dataloader import get_FL_dataloader
from tool.logger import *


def Experiment_Create_dataset(param_dict):
    dataset_name = [i.strip().lower() for i in param_dict['dataset_name'].split(",")]
    data_path = []
    get_dataset = []

    if "CNNDM".lower() in dataset_name:
        # data_path = "./dataset/CNNDM"
        # get_dataset = get_CNNDM_dataset
        data_path.append("./dataset/CNNDM")
        get_dataset.append(get_CNNDM_dataset)

    if "WikiHow".lower() in dataset_name:
        # data_path = "./dataset/WikiHow"
        # get_dataset = get_WikiHow_dataset
        data_path.append("./dataset/WikiHow")
        get_dataset.append(get_WikiHow_dataset)

    if "GovernmentReport".lower() in dataset_name:
        # data_path = "./dataset/GovernmentReport"
        # get_dataset = get_GovernmentReport_dataset
        data_path.append("./dataset/GovernmentReport")
        get_dataset.append(get_GovernmentReport_dataset)

    if "PubMed".lower() in dataset_name:
        # data_path = "./dataset/PubMed"
        # get_dataset = get_PubMed_dataset
        data_path.append("./dataset/PubMed")
        get_dataset.append(get_PubMed_dataset)

    training_dataset = []
    validation_dataset = []
    testing_dataset = []
    for i in range(len(data_path)):
        g = get_dataset[i]
        d = data_path[i]

        # training_dataset.append(g(d, "train", only_one=False))
        # validation_dataset.append(g(d, "valid", only_one=False))
        # testing_dataset.append(g(d, "test", only_one=False))
        training_dataset.append(g(d, "train", only_one=True))
        validation_dataset.append(g(d, "valid", only_one=True))
        testing_dataset += g(d, "test", only_one=True)



    return training_dataset, validation_dataset, testing_dataset


def Experiment_Create_dataloader(param_dict, training_dataset, validation_dataset, testing_dataset, split_strategy="Uniform"):
    num_clients_K = param_dict['num_clients_K']
    batch_size = param_dict['batch_size']

    # 一类数据被存储到list的一个项中
    data_field_number = len(training_dataset)

    if data_field_number == 1:
        training_dataloaders, client_dataset_list = get_FL_dataloader(
            training_dataset[-1], num_clients_K, split_strategy=split_strategy,
            do_train=True, batch_size=batch_size, num_workers=0, do_shuffle=True
        )

        validation_dataloaders, _ = get_FL_dataloader(
            validation_dataset[-1], num_clients_K, split_strategy=split_strategy,
            do_train=True, batch_size=batch_size, num_workers=0, do_shuffle=True
        )

    else:
        training_dataloaders = []
        client_dataset_list = []
        validation_dataloaders = []

        filed_size = [num_clients_K // data_field_number for i in range(data_field_number)]
        filed_size[-1] += num_clients_K % data_field_number

        for i in range(data_field_number):
            td, cd = get_FL_dataloader(
                training_dataset[i], filed_size[i], split_strategy=split_strategy,
                do_train=True, batch_size=batch_size, num_workers=0, do_shuffle=True
            )

            vd, _ = get_FL_dataloader(
                validation_dataset[i], filed_size[i], split_strategy=split_strategy,
                do_train=True, batch_size=batch_size, num_workers=0, do_shuffle=True
            )

            training_dataloaders += td
            client_dataset_list += cd
            validation_dataloaders += vd

    testing_dataloader = get_FL_dataloader(
        testing_dataset, num_clients_K, split_strategy="Uniform",
        do_train=False, batch_size=batch_size, num_workers=0
    )

    return training_dataloaders, validation_dataloaders, client_dataset_list, testing_dataloader


def Experiment_Create_model(param_dict):
    if "BERTSUMEXT".lower() in param_dict['hypothesis'].lower():
        logger.info("Model construction (BERTSUMEXT)")
        model = ExtSummarizer(classifier_type=param_dict['classifier_type'])
    else:
        logger.info("Model construction (AREDSUM)")
        model = None
    model.to(param_dict['device'])
    return model


def Experiment_Reload_model(checkpoint_path):
    model = torch.load(checkpoint_path)
    return model