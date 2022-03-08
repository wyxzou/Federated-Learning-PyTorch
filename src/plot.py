import pickle
import numpy as np
import matplotlib.pyplot as plt



def plot_graphs(x, y, title, xlabel, ylabel, legend_labels):
    plt.figure()
    n = len(x)
    
    for i in range(n):
        plt.plot(x[i], y[i], label=legend_labels[i])
    
    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    # plt.savefig("100 workers")


def plot_iid_sparsetopk_baseline():
    dataset = "mnist"
    model = "mlp"
    epochs = 60
    num_users = 1
    frac = "1.0"
    iid = 1
    local_bs = 10
    optimizer = "sparsetopk"
    lrs = [0.01, 0.005, 0.001] # 0.0005]
    bidirectional = 1
    numbers = [1, 2, 3, 4, 5]

    plot_files(dataset, model, epochs, num_users, frac, iid, local_bs, optimizer, lrs, bidirectional, numbers, "dir1")


def plot_iid_sgd_baseline():
    dataset = "mnist"
    model = "mlp"
    epochs = 60
    num_users = 1
    frac = "1.0"
    iid = 1
    local_bs = 10
    optimizer = "sgd"
    lrs = [0.01, 0.005, 0.001] # 0.0005]
    bidirectional = 1
    numbers = [1, 2, 3]

    plot_files(dataset, model, epochs, num_users, frac, iid, local_bs, optimizer, lrs, bidirectional, numbers, "sgd")


def plot_iid_sparsetopk_baseline():
    dataset = "mnist"
    model = "mlp"
    epochs = 60
    num_users = 1
    frac = "1.0"
    iid = 1
    local_bs = 10
    optimizer = "sparsetopk"
    lrs = [0.01, 0.005, 0.001] # 0.0005]
    bidirectional = 1
    numbers = [1, 2, 3, 4, 5]

    plot_files(dataset, model, epochs, num_users, frac, iid, local_bs, optimizer, lrs, bidirectional, numbers, "dir1")


def plot_users_100_dir_1_sparsetopk():
    dataset = "mnist"
    model = "mlp"
    epochs = 80
    num_users = 100
    frac = "1.0"
    iid = 1
    local_bs = 10
    optimizer = "sparsetopk"
    lrs = [0.01, 0.005]
    bidirectional = 1
    numbers = [1]

    plot_files(dataset, model, epochs, num_users, frac, iid, local_bs, optimizer, lrs, bidirectional, numbers, "bidir_vs_unidir2")


def plot_users_100_dir_1_sparsetopk_batch_64():
    dataset = "mnist"
    model = "mlp"
    epochs = 80
    num_users = 100
    frac = "1.0"
    iid = 1
    local_bs = 64
    optimizer = "sparsetopk"
    lrs = [0.01, 0.005, 0.001]
    bidirectional = 1
    numbers = [1]

    plot_files(dataset, model, epochs, num_users, frac, iid, local_bs, optimizer, lrs, bidirectional, numbers, "bidir_vs_unidir2")


def plot_users_100_dir_0_sparsetopk():
    dataset = "mnist"
    model = "mlp"
    epochs = 80
    num_users = 100
    frac = "1.0"
    iid = 1
    local_bs = 10
    optimizer = "sparsetopk"
    lrs = [0.01, 0.005, 0.001]
    bidirectional = 0
    numbers = [1]

    plot_files(dataset, model, epochs, num_users, frac, iid, local_bs, optimizer, lrs, bidirectional, numbers, "bidir_vs_unidir2")


def plot_users_100_dir_0_sparsetopk_batch_64():
    dataset = "mnist"
    model = "mlp"
    epochs = 80
    num_users = 100
    frac = "1.0"
    iid = 1
    local_bs = 64
    optimizer = "sparsetopk"
    lrs = [0.01, 0.005, 0.001]
    bidirectional = 0
    numbers = [1]

    plot_files(dataset, model, epochs, num_users, frac, iid, local_bs, optimizer, lrs, bidirectional, numbers, "bidir_vs_unidir2")

def plot_users_100_sgd_batch_64():
    dataset = "mnist"
    model = "mlp"
    epochs = 80
    num_users = 100
    frac = "1.0"
    iid = 1
    local_bs = 64
    optimizer = "sgd"
    lrs = [0.01, 0.005, 0.001]
    bidirectional = 0
    numbers = [1]

    plot_files(dataset, model, epochs, num_users, frac, iid, local_bs, optimizer, lrs, bidirectional, numbers, "bidir_vs_unidir2")


def plot_users_100_sgd():
    dataset = "mnist"
    model = "mlp"
    epochs = 80
    num_users = 100
    frac = "1.0"
    iid = 1
    local_bs = 10
    optimizer = "sgd"
    lrs = [0.01, 0.005, 0.001]
    bidirectional = 0
    numbers = [1]

    plot_files(dataset, model, epochs, num_users, frac, iid, local_bs, optimizer, lrs, bidirectional, numbers, "bidir_vs_unidir2")



def compare_baselines():
    dataset = "mnist"
    model = "mlp"
    epochs = 60
    num_users = 1
    frac = "1.0"
    iid = 1
    local_bs = 10

    all_experiments = []
    experiments1 = []
    for number in [1, 2, 3]:
        file_name1 = '../tuning/dir1/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl'  \
            .format(dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", 0.005, 1, number)

        with open(file_name1, 'rb') as pickle_file:
            experiments1.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments1)[:, 3], axis=0))

    experiments2 = []
    for number in [1, 2, 3]:
        file_name2 = '../tuning/sgd/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sgd", 0.01, 1, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))

    plot_graphs([list(range(epochs))] * 2, all_experiments, "", "Epoch", "Accuracy", ["sparsetopk", "sgd"])



def sgd_results():
    dataset = "mnist"
    model = "mlp"
    epochs = 80
    num_users = 20
    frac = "1.0"
    iid = 1
    local_bs = 10

    all_experiments = []
    experiments1 = []
    for number in [1, 2, 3]:
        file_name1 = '{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl'  \
            .format(dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sgd", 0.01, 0, number)

        with open(file_name1, 'rb') as pickle_file:
            experiments1.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments1)[:, 3], axis=0))


    plot_graphs([list(range(epochs))], all_experiments, "", "Epoch", "Accuracy", ["sgd"])


def compare_mnist():
    dataset = "mnist"
    model = "mlp"
    epochs = 100
    num_users = 20
    frac = "1.0"
    iid = 1
    local_bs = 10

    all_experiments = []
    experiments1 = []
    for number in [1, 2, 3, 4, 5]:
        file_name1 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl'  \
            .format(dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", 0.01, 0, number)

        with open(file_name1, 'rb') as pickle_file:
            experiments1.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments1)[:, 3], axis=0))


    experiments3 = []
    for number in [1, 2, 3, 4, 5]:
        file_name3 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl'  \
            .format(dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", 0.01, 1, number)

        with open(file_name3, 'rb') as pickle_file:
            experiments3.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments3)[:, 3], axis=0))


    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 1, frac, iid,
                    local_bs, "sgd", 0.01, 0, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    plot_graphs([list(range(epochs))] * 3, all_experiments, "", "Epoch", "Accuracy", ["sparsetopk unidirectional", "sparsetopk bidirectional", "sgd"])



def compare_worker():
    dataset = "mnist"
    model = "mlp"
    epochs = 80
    num_users = 100
    frac = "1.0"
    iid = 1
    local_bs = 10

    all_experiments = []
    experiments1 = []
    for number in [1, 2, 3]:
        file_name1 = '../100baseline/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl'  \
            .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sgd", 0.01, 0, number)

        with open(file_name1, 'rb') as pickle_file:
            experiments1.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments1)[:, 3], axis=0))


    experiments3 = []
    for number in [2, 3]:
        file_name3 = '../100baseline/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl'  \
            .format(dataset, model, epochs, 100, frac, iid,
                    local_bs, "sgd", 0.01, 0, number)

        with open(file_name3, 'rb') as pickle_file:
            experiments3.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments3)[:, 3], axis=0))


    # experiments2 = []
    # for number in [1]:
    #     file_name2 = '../bidir_vs_unidir2/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
    #             .format(dataset, model, epochs, num_users, frac, iid,
    #                 local_bs, "sgd", 0.005, 0, number)

    #     with open(file_name2, 'rb') as pickle_file:
    #         experiments2.append(pickle.load(pickle_file))

    # all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    plot_graphs([list(range(epochs))] * 2, all_experiments, "", "Epoch", "Accuracy", ["20 users", "100 users"])




def compare_batch_size():
    dataset = "mnist"
    model = "mlp"
    epochs = 80
    num_users = 100
    frac = "1.0"
    iid = 1
    local_bs = 10

    all_experiments = []
    experiments1 = []
    for number in [1]:
        file_name1 = '../bidir_vs_unidir2/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl'  \
            .format(dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", 0.005, 0, number)

        with open(file_name1, 'rb') as pickle_file:
            experiments1.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments1)[:, 3], axis=0))


    experiments3 = []
    for number in [1]:
        file_name3 = '../bidir_vs_unidir2/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl'  \
            .format(dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", 0.005, 1, number)

        with open(file_name3, 'rb') as pickle_file:
            experiments3.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments3)[:, 3], axis=0))


    experiments2 = []
    for number in [1]:
        file_name2 = '../bidir_vs_unidir2/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sgd", 0.005, 0, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    plot_graphs([list(range(epochs))] * 3, all_experiments, "", "Epoch", "Accuracy", ["sparsetopk", "sparsetopk unidirectional", "sgd"])


def train_learning_rate():
    dataset = "fmnist"
    model = "mlp"
    epochs = 100
    num_users = 100
    frac = "1.0"
    iid = 1
    local_bs = 10

    experiments1 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 1, frac, iid,
                    local_bs, "sparsetopk", 0.002, 1, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments1.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))

    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 1, frac, iid,
                    local_bs, "sparsetopk", 0.005, 1, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    experiments3 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 1, frac, iid,
                    local_bs, "sparsetopk", 0.008, 1, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments3.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments3)[:, 3], axis=0))

    experiments4 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 1, frac, iid,
                    local_bs, "sparsetopk", 0.01, 1, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments4.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments4)[:, 3], axis=0))

    plot_graphs([list(range(epochs))] * 4, all_experiments, "", "Epoch", "Accuracy", ["0.001", "0.002", "0.008", "0.01"])



def fminist_baseline_unidirectional():
    dataset = "fmnist"
    model = "mlp"
    epochs = 100
    num_users = 20
    frac = "1.0"
    iid = 1
    local_bs = 10

    all_experiments = []
    experiments1 = []
    for number in [1]:
        file_name1 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl'  \
            .format(dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", 0.01, 0, number)

        with open(file_name1, 'rb') as pickle_file:
            experiments1.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments1)[:, 3], axis=0))


    experiments3 = []
    for number in [1]:
        file_name3 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl'  \
            .format(dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", 0.008, 0, number)

        with open(file_name3, 'rb') as pickle_file:
            experiments3.append(pickle.load(pickle_file))


    all_experiments.append(np.mean(np.array(experiments3)[:, 3], axis=0))


    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", 0.005, 0, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))



    experiments4 = []
    for number in [1]:
        file_name4 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", 0.002, 0, number)

        with open(file_name4, 'rb') as pickle_file:
            experiments4.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments4)[:, 3], axis=0))


    plot_graphs([list(range(epochs))] * 4, all_experiments, "Tuning Unidirectional TopK on Learning Rates", "Epoch", "Accuracy", ["0.01", "0.008", "0.005", "0.002"])


def sgd_baseline_one_worker():
    dataset = "fmnist"
    model = "mlp"
    epochs = 100
    num_users = 20
    frac = "1.0"
    iid = 1
    local_bs = 10
    all_experiments = []
    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 1, frac, iid,
                    local_bs, "sgd", 0.02, 0, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))

    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 1, frac, iid,
                    local_bs, "sgd", 0.01, 0, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 1, frac, iid,
                    local_bs, "sgd", 0.002, 0, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 1, frac, iid,
                    local_bs, "sgd", 0.005, 0, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    plot_graphs([list(range(epochs))] * 4, all_experiments, "", "Epoch", "Accuracy", ["0.02", "0.01", "0.005", "0.002"])



def sgd_baseline_20_worker():
    dataset = "fmnist"
    model = "mlp"
    epochs = 100
    num_users = 20
    frac = "1.0"
    iid = 1
    local_bs = 10
    all_experiments = []
    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sgd", 0.02, 0, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))

    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sgd", 0.02, 0, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    plot_graphs([list(range(epochs))] * 2, all_experiments, "Tuning SGD on Learning Rate (Twenty Workers)", "Epoch", "Accuracy", ["0.02", "0.01"])



def plot_sparsity():
    dataset = "fmnist"
    model = "mlp"
    epochs = 100
    num_users = 20
    frac = "1.0"
    iid = 1
    local_bs = 10
    topk = 0.001
    lr = 0.01

    all_experiments = []

    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", lr, 1, topk, 0.01, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))



    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", lr, 1, topk, 0.1, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", lr, 1, topk, 0.1, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    


    plot_graphs([list(range(epochs))] * 2, all_experiments, "Tuning SGD on Learning Rate (Twenty Workers)", "Epoch", "Accuracy", ["0.02", "0.01"])




def bidirectional_sparsetopk():
    dataset = "fmnist"
    model = "mlp"
    epochs = 100
    num_users = 20
    frac = "1.0"
    iid = 1
    local_bs = 10
    topk = 0.001
    topk_d = 0.001
    all_experiments = []


    # experiments2 = []
    # for number in [1]:
    #     file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
    #             .format(dataset, model, epochs, 20, frac, iid,
    #                 local_bs, "sparsetopk", 0.01, 1, topk, topk_d, number)

    #     with open(file_name2, 'rb') as pickle_file:
    #         experiments2.append(pickle.load(pickle_file))

    # all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    experiments2 = []
    for number in [1]:
        file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.02, 1, topk, topk_d, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    experiments2 = []
    for number in [1]:
        file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.05, 1, topk, topk_d, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    experiments2 = []
    for number in [1]:
        file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.06, 1, topk, topk_d, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))

    # experiments2 = []
    # for number in [1]:
    #     file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
    #             .format(dataset, model, epochs, 20, frac, iid,
    #                 local_bs, "sparsetopk", 0.04, 1, number)

    #     with open(file_name2, 'rb') as pickle_file:
    #         experiments2.append(pickle.load(pickle_file))

    # all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))



    # experiments2 = []
    # for number in [1]:
    #     file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
    #             .format(dataset, model, epochs, 20, frac, iid,
    #                 local_bs, "sparsetopk", 0.05, 1, number)

    #     with open(file_name2, 'rb') as pickle_file:
    #         experiments2.append(pickle.load(pickle_file))

    # all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))



    plot_graphs([list(range(epochs))] * 3, all_experiments, "Tuning Bidirectional TopK on Learning Rate (Twenty Workers)", "Epoch", "Accuracy", ["0.02", "0.05", "0.06"])





def bidirectional_sparsetopk_different_sparsity():
    dataset = "fmnist"
    model = "mlp"
    epochs = 100
    num_users = 20
    frac = "1.0"
    iid = 1
    local_bs = 10
    all_experiments = []
    topk = 0.001

    experiments2 = []
    for number in [1]:
        file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.01, 1, topk, 0.01, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))



    experiments2 = []
    for number in [1]:
        file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.01, 1, topk, 0.1, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    experiments2 = []
    for number in [1]:
        file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.01, 1, topk, 0.25, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    experiments2 = []
    for number in [1]:
        file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.01, 1, topk, 0.5, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    experiments2 = []
    for number in [1]:
        file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.01, 1, topk, 0.75, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))



    experiments2 = []
    for number in [1]:
        file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.01, 1, topk, "1.0", number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))



    plot_graphs([list(range(epochs))] * 6, all_experiments, "Effect of Different Sparsity on Bidirectional TopK", "Epoch", "Accuracy", ["0.1", "0.01", "0.25", "0.5", "0.75", "1"])





def unidirectional_sparsetopk_different_sparsity():
    dataset = "fmnist"
    model = "mlp"
    epochs = 100
    num_users = 20
    frac = "1.0"
    iid = 1
    local_bs = 10
    all_experiments = []


    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sgd", 0.01, 0, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.01, 0, 0.1, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))



    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.01, 0, 0.01, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.01, 0, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))




    plot_graphs([list(range(epochs))] * 4, all_experiments, "Effect of Sparsity on Unidirectional TopK", "Epoch", "Accuracy", ["1", "0.1", "0.01", "0.001"])




def compare1_to_20_worker():
    dataset = "fmnist"
    model = "mlp"
    epochs = 100
    num_users = 20
    frac = "1.0"
    iid = 1
    local_bs = 10
    all_experiments = []
    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 1, frac, iid,
                    local_bs, "sgd", 0.01, 0, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))

    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sgd", 0.02, 0, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    plot_graphs([list(range(epochs))] * 2, all_experiments, "Comparison Between Different Number of Workers", "Epoch", "Accuracy", ["1 worker (lr = 0.01)", "20 worker (lr = 0.02)"])


def compare_fmnist():
    dataset = "fmnist"
    model = "mlp"
    epochs = 100
    num_users = 20
    frac = "1.0"
    iid = 1
    local_bs = 10
    topk = 0.001
    topk_d = 0.001

    all_experiments = []
    experiments1 = []
    for number in [1]:
        file_name1 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl'  \
            .format(dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", 0.01, 0, number)

        with open(file_name1, 'rb') as pickle_file:
            experiments1.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments1)[:, 3], axis=0))


    experiments2 = []
    for number in [1]:
        file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", 0.06, 1, topk, topk_d, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sgd", 0.02, 0, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 1, frac, iid,
                    local_bs, "sgd", 0.01, 0, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    plot_graphs([list(range(epochs))] * 3, all_experiments, "", "Epoch", "Accuracy", ["sparsetopk unidirectional", "sparsetopk bidirectional", "sgd"])


def plot_compression():
    dataset = "fmnist"
    model = "mlp"
    epochs = 100
    num_users = 100
    frac = "1.0"
    iid = 1
    local_bs = 10

    experiments1 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 1, frac, iid,
                    local_bs, "sparsetopk", 0.002, 1, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments1.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))

    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 1, frac, iid,
                    local_bs, "sparsetopk", 0.005, 1, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    experiments3 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 1, frac, iid,
                    local_bs, "sparsetopk", 0.008, 1, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments3.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments3)[:, 3], axis=0))

    experiments4 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 1, frac, iid,
                    local_bs, "sparsetopk", 0.01, 1, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments4.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments4)[:, 3], axis=0))

    plot_graphs([list(range(epochs))] * 4, all_experiments, "", "Epoch", "Accuracy", ["0.001", "0.002", "0.008", "0.01"])



def plot_xi_values():
    dataset = "fmnist"
    model = "mlp"
    epochs = 20
    num_users = 100
    frac = "1.0"
    iid = 1
    local_bs = 10
    topk = 0.001
    topk_d = 0.001

    all_experiments = []
    experiments2 = []
    for number in [1]:
        file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.06, 1, topk, topk_d, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 4], axis=0))



    experiments2 = []
    for number in [1]:
        file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.01, 0, topk, topk_d, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 4], axis=0))


    batches = len(all_experiments[0])
    plot_graphs([list(range(batches))] * 2, all_experiments, "", "Batch", r"$\xi$", ["bidirectional", "unidirectional"])



def plot_xi_values():
    dataset = "fmnist"
    model = "mlp"
    epochs = 20
    num_users = 100
    frac = "1.0"
    iid = 1
    local_bs = 10
    topk = 0.001
    topk_d = 0.001

    all_experiments = []
    experiments2 = []
    for number in [1]:
        file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.06, 1, topk, topk_d, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 4], axis=0))



    experiments2 = []
    for number in [1]:
        file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.01, 0, topk, topk_d, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 4], axis=0))


    batches = len(all_experiments[0])
    plot_graphs([list(range(batches))] * 2, all_experiments, "", "Batch", r"$\xi$", ["bidirectional", "unidirectional"])



def plot_compression_values():
    dataset = "fmnist"
    model = "mlp"
    epochs = 20
    num_users = 100
    frac = "1.0"
    iid = 1
    local_bs = 10
    topk = 0.001
    topk_d = 0.001

    all_experiments = []
    experiments2 = []
    for number in [1]:
        file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.06, 1, topk, topk_d, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(1 - np.mean(np.array(experiments2)[:, 5], axis=0))



    batches = len(all_experiments[0])
    plot_graphs([list(range(batches))] * 1, all_experiments, "Percentage of Gradient Compressed Donwstream ", "Batch", "", ["downstream"])





def plot_compression_values_upstream():
    dataset = "fmnist"
    model = "mlp"
    epochs = 20
    num_users = 100
    frac = "1.0"
    iid = 1
    local_bs = 10
    topk = 0.001
    topk_d = 0.001

    all_experiments = []
    experiments2 = []
    for number in [1]:
        file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.01, 0, topk, topk_d, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(1 - np.mean(np.array(experiments2)[:, 5], axis=0))



    batches = len(all_experiments[0])
    plot_graphs([list(range(batches))] * 1, all_experiments, "Percentage of Gradient Compressed Downstream ", "Batch", "", ["upstream"])




def plot_delta():
    dataset = "fmnist"
    model = "mlp"
    epochs = 20
    num_users = 100
    frac = "1.0"
    iid = 1
    local_bs = 10
    topk = 0.001
    topk_d = 0.001

    all_experiments = []
    experiments2 = []
    for number in [1]:
        file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.01, 0, topk, topk_d, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 6], axis=0))


    experiments2 = []
    for number in [1]:
        file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.06, 1, topk, topk_d, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 6], axis=0)) 



    experiments2 = []
    for number in [1]:
        file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.06, 1, topk, topk_d, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 7], axis=0))


    batches = len(all_experiments[0])
    plot_graphs([list(range(batches))] * 3, all_experiments, "", "Batch", r"$1 - \gamma$", ["unidirectional upstream", "bidirectional upstream", "bidirectional downstream"])



def plot_cnn():
    dataset = "fmnist"
    model = "cnn"
    epochs = 20
    num_users = 100
    frac = "1.0"
    iid = 1
    local_bs = 10
    topk = 0.001
    topk_d = 0.001

    all_experiments = []
    experiments2 = []
    for number in [1]:
        file_name2 = '../save/bidir_fix/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.01, 1, topk, topk_d, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))


    experiments2 = []
    for number in [1]:
        file_name2 = '../save/inter_batch_communication/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, 20, frac, iid,
                    local_bs, "sparsetopk", 0.01, 0, topk, topk_d, number)

        with open(file_name2, 'rb') as pickle_file:
            experiments2.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments2)[:, 3], axis=0))

    plot_graphs([list(range(epochs))] * 2, all_experiments, "CNN; 20 Users", "Epoch", "Accuracy", ["bidirectional", "unidirectional"])


def plot_files(dataset, model, epochs, num_users, frac, iid, local_bs, optimizer, lrs, bidirectional, numbers, folder):
    avg_experiments = []
    for lr in lrs:
        experiments = []
        for number in numbers:
            file_name = '../{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_NUM[{}].pkl' \
                .format(folder, dataset, model, epochs, num_users, frac, iid,
                       local_bs, optimizer, lr, bidirectional, number)

            
            with open(file_name, 'rb') as pickle_file:
                experiments.append(pickle.load(pickle_file))

        avg_experiments.append(np.mean(np.array(experiments)[:, 3], axis=0))

    plot_graphs([list(range(epochs))] * len(lrs), avg_experiments, "Percentage of Gradient Compressed Upstream", "Epoch", "Accuracy", lrs)


if __name__ == "__main__":
    bidirectional_sparsetopk_different_sparsity()
    # compare_fmnist()
    # train_learning_rate()
    # plot_users_100_dir_1_sparsetopk_batch_64()
    # plot_users_100_sgd_batch_64()
    # plot_users_100_sgd()
    # plot_users_100_dir_1_sparsetopk()
    # compare_results()


    