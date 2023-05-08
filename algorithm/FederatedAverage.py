import torch
import numpy as np
import copy

from tool.logger import *
from tool.utils import get_parameters, set_parameters
from algorithm.Optimizers import BERTSUMEXT_Optimizer


def client_selection(client_num, fraction, dataset_size, client_dataset_size_list, drop_rate, style="FedAvg"):
    assert sum(client_dataset_size_list) == dataset_size
    idxs_users = [0]

    selected_num = max(int(fraction * client_num), 1)
    if float(drop_rate) != 0:
        drop_num = max(int(selected_num * drop_rate), 1)
        selected_num -= drop_num

    if style == "FedAvg":
        idxs_users = np.random.choice(
            a=range(client_num),
            size=selected_num,
            replace=False,
            p=[float(i / dataset_size) for i in client_dataset_size_list]
        )

    return idxs_users


# Federated Average with BERTSUM
def Fed_AVG_BERTSUMEXT(device,
                       global_model,
                       algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
                       training_dataloaders,
                       training_dataset,
                       client_dataset_list):
    # training_dataset_size = len(training_dataset)
    training_dataset_size = sum(len(i) for i in training_dataset)
    client_datasets_size_list = [len(item) for item in client_dataset_list]

    # Training process
    logger.info("Training process")

    # Parameter Initialization
    global_model.train()
    local_model_list = [copy.deepcopy(global_model) for i in range(num_clients_K)]

    criterion = torch.nn.BCELoss(reduction='none')
    for iter_t in range(algorithm_epoch_T):
        # Simulate Client Parallel
        for i in range(num_clients_K):
            model = local_model_list[i]
            model.train()
            model.zero_grad()
            model.to(device)

            optimizer = BERTSUMEXT_Optimizer(
                method="adam", learning_rate=1, max_grad_norm=0,
                beta1=0.9, beta2=0.999,
                decay_method='noam',
                warmup_steps=8000)
            optimizer.set_parameters(list(model.named_parameters()))

            # local option
            client_i_dataloader = training_dataloaders[i]

            logger.info(f"########## Algorithm Epoch: {iter_t + 1} / {algorithm_epoch_T}; "
                        f"Client: {i + 1} / {num_clients_K};  ##########")

            for batch_index, batch in enumerate(client_i_dataloader):
                model.zero_grad()
                src = batch['src'].to(device)
                labels = batch['src_sent_labels'].to(device)
                segs = batch['segs'].to(device)
                clss = batch['clss'].to(device)
                mask = batch['mask_src'].to(device)
                mask_cls = batch['mask_cls'].to(device)

                sent_scores, mask = model(src, segs, clss, mask, mask_cls)
                loss = criterion(sent_scores, labels.float())
                loss = (loss * mask.float()).sum()
                logger.info(f"########## Algorithm Epoch: {iter_t + 1} / {algorithm_epoch_T}; "
                            f"Client: {i + 1} / {num_clients_K}; "
                            f"Batch: {batch_index}; Loss: {loss.data} ##########")
                (loss / loss.numel()).backward()
                optimizer.step()

            # Upgrade the local model list
            local_model_list[i] = model

        # Communicate
        if (iter_t + 1) % communication_round_I == 0:
            logger.info(f"********** Communicate: {(iter_t + 1) / communication_round_I} **********")
            # Client selection
            logger.info(f"********** Client selection **********")
            idxs_users = client_selection(
                client_num=num_clients_K,
                fraction=FL_fraction,
                dataset_size=training_dataset_size,
                client_dataset_size_list=client_datasets_size_list,
                drop_rate=FL_drop_rate,
                style="FedAvg",
            )
            logger.info(f"********** Select client list: {idxs_users} **********")

            # Global operation
            logger.info("********** Parameter aggregation **********")
            theta_list = []
            for id in idxs_users:
                selected_model = local_model_list[id]
                theta_list.append(get_parameters(selected_model))
                # theta_list.append(get_parameters(selected_model.ext_layer))

            theta_list = np.array(theta_list, dtype=object)
            theta_avg = np.mean(theta_list, 0).tolist()
            set_parameters(global_model, theta_avg)

            # Parameter Distribution
            logger.info("********** Parameter distribution **********")
            local_model_list = [copy.deepcopy(global_model) for i in range(num_clients_K)]

    logger.info("Training finish, return global model and local model list")
    return global_model, local_model_list
