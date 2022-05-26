import pdb
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def batch_to_epoch(arr, num, ignore_first=False):
    if len(arr) == 0:
        return []

    i = 0
    epoch_vals = []
    while i + num <= len(arr):
        if ignore_first:
            if i == 0:
                epoch_vals.append(max(arr[i+1: i+num]))
            else:
                epoch_vals.append(max(arr[i: i+num]))
        else:
            epoch_vals.append(max(arr[i: i+num]))
        i = i + num

    return epoch_vals




def batch_to_epoch_avg(arr, num, ignore_first=False):
    if len(arr) == 0:
        return []
    i = 0
    epoch_vals = []
    while i + num <= len(arr):
        if ignore_first:
            if i == 0:
                epoch_vals.append(sum(arr[i + 1: i+num]) / (num - 1))
            else:
                epoch_vals.append(sum(arr[i + 1: i+num]) / num)
        else:
            epoch_vals.append(sum(arr[i + 1: i+num]) / num)
        i = i + num

    return epoch_vals


def batch_to_epoch_min(arr, n):
    i = 0
    epoch_vals = []
    while i + n <= len(arr):
        epoch_vals.append(min(arr[i: i+n]))
        i = i + n

    return epoch_vals


def get_values(lrs, num_users=20, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [1], index=4):
    frac = "1.0"
    iid = 1
    topk = 0.001
    topk_d = 0.001

    all_experiments = []
    experiments = []
    for number in numbers:
        file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", lrs[0], 0, topk, topk_d, number)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))
    xi_values = np.mean(np.array(experiments)[:, index], axis=0)
    all_experiments.append(xi_values)

    experiments = []
    for number in numbers:
        file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", lrs[1], 1, topk, topk_d, number)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))

    xi_values = np.mean(np.array(experiments)[:, index], axis=0)
    all_experiments.append(xi_values)

    epochs = max(len(all_experiments[0]), len(all_experiments[1]))

    return epochs, all_experiments



def get_downlink(lrs, num_users=20, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [1], index=4, ignore_first=False, topk_d=0.001):
    frac = "1.0"
    iid = 1
    topk = 0.001


    experiments = []
    for number in numbers:
        file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", lrs[1], 1, topk, topk_d, number)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))

    xi_values = np.mean(np.array(experiments)[:, index], axis=0)
    d = len(xi_values) // 100
    xi_values = batch_to_epoch(xi_values, d, ignore_first)
    epochs = len(xi_values)

    return epochs, xi_values



def get_downlink_batches(lrs, num_users=20, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [1], index=4, ignore_first=False, topk_d=0.001):
    frac = "1.0"
    iid = 1
    topk = 0.001


    experiments = []
    for number in numbers:
        file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", lrs[1], 1, topk, topk_d, number)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))

    xi_values = np.mean(np.array(experiments)[:, index], axis=0)
    # all_experiments.append(xi_values)

    epochs = len(xi_values)

    return epochs, xi_values



def get_xi_values(lrs, num_users=20, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [1], index=4, ignore_first=False, topk_d=0.001, iid = 1):
    frac = "1.0"
    
    topk = 0.001

    all_experiments = []
    experiments = []
    for number in numbers:
        file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", lrs[0], 0, topk, topk_d, number)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))
    xi_values = np.mean(np.array(experiments)[:, index], axis=0)
    d = len(xi_values) // 100
    xi_values = batch_to_epoch(xi_values, d, ignore_first)
    all_experiments.append(xi_values)

    experiments = []
    for number in numbers:
        file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", lrs[1], 1, topk, topk_d, number)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))

    xi_values = np.mean(np.array(experiments)[:, index], axis=0)
    d = len(xi_values) // 100
    xi_values = batch_to_epoch(xi_values, d, ignore_first)
    all_experiments.append(xi_values)

    epochs = len(all_experiments[0])

    return epochs, all_experiments


def get_side_values(lr, num_users=20, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [1], direction=0, iid = 1):
    frac = "1.0"
   
    topk = 0.001
    topk_d = 0.001

    all_experiments = []
    experiments = []
    for number in numbers:
        file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", lr, direction, topk, topk_d, number)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))
    xi_values_lhs = np.mean(np.array(experiments)[:, 9], axis=0)
    d = len(xi_values_lhs) // 100
    xi_values_lhs = batch_to_epoch(xi_values_lhs, d)
    all_experiments.append(xi_values_lhs)


    xi_values_rhs = np.mean(np.array(experiments)[:, 10], axis=0)
    d = len(xi_values_rhs) // 100
    xi_values_rhs = batch_to_epoch_min(xi_values_rhs, d)
    all_experiments.append(xi_values_rhs)

    epochs = len(all_experiments[0])

    return epochs, all_experiments


def get_distance(lrs, num_users=20, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers=[4], exception=False):
    frac = "1.0"
    iid = 1
    topk = 0.001
    topk_d = 0.001

    all_experiments = []

    experiments = []
    for number in numbers:
        file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", lrs[1], 1, topk, topk_d, number)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))

    lhs_values = np.mean(np.array(experiments)[:, 12], axis=0)
    gradient = np.mean(np.array(experiments)[:, 10], axis=0)

    a = lhs_values/gradient
    d = len(a) // 100
    xi_values = batch_to_epoch(a, d)

    all_experiments.append(xi_values)

    batches = len(all_experiments[0])

    return all_experiments



def get_side_values_batches(lr, num_users=20, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [1], direction=0):
    frac = "1.0"
    iid = 1
    topk = 0.001
    topk_d = 0.001

    all_experiments = []
    experiments = []
    for number in numbers:
        file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", lr, direction, topk, topk_d, number)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))
    xi_values_lhs = np.mean(np.array(experiments)[:, 9], axis=0)
    all_experiments.append(xi_values_lhs)


    xi_values_rhs = np.mean(np.array(experiments)[:, 10], axis=0)
    all_experiments.append(xi_values_rhs)

    epochs = len(all_experiments[0])

    return epochs, all_experiments


def get_model_results(lrs, num_users=20, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [1], index_comparison=3, sgddir=0):
    frac = "1.0"
    iid = 1
    topk = 0.001
    topk_d = 0.001

    all_experiments = []

    experiments = []
    for num in [1]:
        file_name = '../save/{}-{}-tuning/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", lrs[0], 0, topk, topk_d, num)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file)[index_comparison])

    all_experiments.append(np.average(np.array(experiments), axis=0))

    experiments = []
    for num in [1]:
        file_name = '../save/{}-{}-tuning/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", lrs[1], 1, topk, topk_d, num)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file)[index_comparison])

    all_experiments.append(np.average(np.array(experiments), axis=0))


    experiments = []
    for num in [1]:
        file_name = '../save/{}-{}-tuning/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sgd", lrs[2], sgddir, topk, topk_d, num)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file)[index_comparison])

    all_experiments.append(np.average(np.array(experiments), axis=0))

    epochs = len(all_experiments[0])

    return epochs, all_experiments
