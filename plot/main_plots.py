import pdb
import pickle
import numpy as np
import matplotlib.pyplot as plt

from retrieval import get_side_values, get_side_values_batches, get_values, get_downlink


def plot_graphs(x, y, title, xlabel, ylabel, legend_labels, savefile, rotation=90):
    plt.figure(figsize=(4, 3))
    n = len(x)
    
    for i in range(n):
        plt.plot(x[i], y[i], label=legend_labels[i])
    
    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel, labelpad=10, rotation=rotation)
    plt.legend()
    plt.grid()
    plt.savefig(savefile  + ".pdf", bbox_inches='tight')


def plot_graphs_wo_legend(x, y, title, xlabel, ylabel, savefile, rotation=90):
    plt.figure()
    n = len(x)
    
    for i in range(n):
        plt.plot(x[i], y[i])
    

    plt.xlabel(xlabel)
    plt.ylabel(ylabel, labelpad=10, rotation=rotation)
    plt.grid()
    # plt.show()
    plt.savefig(savefile  + ".pdf")


def comparison_between_workers(x, y1, y2, y3, y4):
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    
    plot_subplot(axs[0, 0], x, y1, "10 workers", "epochs", ["unidirectional", "bidirectional", "sgd"])


def plot_subplot(ax, x, y, title, xlabel, ylabel, legend_labels):
    n = len(x)
    
    for i in range(n):
        ax.plot(x[i], y[i], label=legend_labels[i])
    
    ax.set_title(title)

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend()
    # plt.savefig(savefile  + ".png")



def tuning(num_users, direction, lrs, model="mlp", dataset="mnist", epochs=100, local_bs=10, iid=1):
    frac = "1.0"
    topk = 0.001
    topk_d = 0.001

    all_experiments = []

    for lr in lrs: 
        experiments = []
        file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sgd", lr, direction, topk, topk_d, 1)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))

        all_experiments.append(np.mean(np.array(experiments)[:, 3], axis=0))


    filename = '{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}]' \
                .format(dataset, model, epochs, num_users, frac, iid, local_bs, "sgd", direction, topk, topk_d, 1)

    savefile = '../save/plots/tuning/{}/'.format(dataset) + filename

    plot_graphs([list(range(epochs))] * len(lrs), all_experiments, "Tuning Learning Rate", "Epoch", "Accuracy", lrs, savefile)



def sparse_tuning(num_users, direction, lrs, model="mlp", dataset = "mnist", epochs=100, local_bs = 10, folder="tuning", iid = 1, number=1):
    frac = "1.0"
    topk = 0.001
    topk_d = 0.001

    all_experiments = []


    for lr in lrs: 
        filename = '{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, num_users, frac, iid, local_bs, "sparsetopk", lr, direction, topk, topk_d, number)
        
        experiments = []
        
        filepath = '../save/{}-{}/'.format(dataset, model) + filename

        with open(filepath, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))

        all_experiments.append(np.mean(np.array(experiments)[:, 3], axis=0))

    filename = '{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}]' \
                .format(dataset, model, epochs, num_users, frac, iid, local_bs, "sparsetopk", direction, topk, topk_d, number)

    savefile = '../save/plots/tuning/{}/'.format(dataset) + filename

    plot_graphs([list(range(epochs))] * len(lrs), all_experiments, "", "Epoch", "Accuracy", lrs, savefile, rotation=90)



def comparison(num_users, lrs, model="mlp", dataset = "mnist", epochs=100, local_bs=10, index_comparison=3, ylabel="Loss", iid=1, nums=[1], savedirectory= "../save/plots/tuning"):
    frac = "1.0"
    topk = 0.001
    topk_d = 0.001


    all_experiments = []

    experiments = []
    for num in nums:
        file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", lrs[0], 0, topk, topk_d, num)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file)[index_comparison])

    # pdb.set_trace()
    all_experiments.append(np.average(np.array(experiments), axis=0))

    experiments = []
    for num in nums:
        file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", lrs[1], 1, topk, topk_d, num)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file)[index_comparison])

    all_experiments.append(np.average(np.array(experiments), axis=0))


    experiments = []
    for num in nums:
        file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sgd", lrs[2], 0, topk, topk_d, num)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file)[index_comparison])

    all_experiments.append(np.average(np.array(experiments), axis=0))
        

    filename = model + "_"+ str(num_users) + "_comparison_" + str(index_comparison) 

    savefile = '{}/{}/'.format(savedirectory, dataset) + filename

    plot_graphs([list(range(epochs))] * len(lrs), all_experiments, "", "Epoch", ylabel, ["unidirectional", "bidirectional", "sgd"], savefile)



def batch_to_epoch_avg(arr, n):
    i = 0
    epoch_vals = []
    while i + n < len(arr):
        epoch_vals.append(sum(arr[i: i+n])/n)
        i = i + n

    return epoch_vals

def batch_to_epoch(arr, n):
    i = 0
    epoch_vals = []
    while i + n < len(arr):
        epoch_vals.append(max(arr[i: i+n]))
        i = i + n

    return epoch_vals


def plot_adjusted_values(lrs, num_users=20, epochs=100, model = "mlp", dataset = "mnist", local_bs=10):
    frac = "1.0"
    iid = 1
    topk = 0.001
    topk_d = 0.001


    all_experiments = []
    experiments = []
    for number in [1]:
        file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", lrs[0], 0, topk, topk_d, number)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))
    xi_values = np.mean(np.array(experiments)[:, 11], axis=0)
    d = len(xi_values) // 100
    # xi_values = batch_to_epoch(xi_values, d)
    print(xi_values)
    all_experiments.append(xi_values)
    print(np.max(xi_values))
    print(np.mean(xi_values))


    experiments = []
    for number in [1]:
        file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", lrs[1], 1, topk, topk_d, number)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))

    xi_values = np.mean(np.array(experiments)[:, 11], axis=0)
    d = len(xi_values) // 100
    # xi_values = batch_to_epoch(xi_values, d)
    all_experiments.append(xi_values)

    print(np.max(xi_values))
    print(np.mean(xi_values))
    batches = len(all_experiments[0])

    title = "Number of Workers: " + str(num_users)
    filename = "adjusted_comparison_" + dataset + "_" + model + "_" + str(num_users)

    savefile = '../save/plots/tuning/{}/'.format(dataset) + filename

    plot_graphs([list(range(batches))] * 2, all_experiments, title, "Epoch", "", ["unidirectional", "bidirectional"], savefile)


def plot_xi_values(lrs, num_users=20, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, rotation=0, directory="thesis_plots"):
    frac = "1.0"
    iid = 1
    topk = 0.001
    topk_d = 0.001

    all_experiments = []
    experiments = []
    for number in [1]:
        file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", lrs[0], 0, topk, topk_d, number)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))
    xi_values = np.mean(np.array(experiments)[:, 4], axis=0)
    d = len(xi_values) // 100
    xi_values = batch_to_epoch(xi_values, d)
    print(xi_values)
    all_experiments.append(xi_values)
    print(np.max(xi_values))
    print(np.mean(xi_values))


    experiments = []
    for number in [1]:
        file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", lrs[1], 1, topk, topk_d, number)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))

    xi_values = np.mean(np.array(experiments)[:, 4], axis=0)
    d = len(xi_values) // 100
    xi_values = batch_to_epoch(xi_values, d)
    all_experiments.append(xi_values)

    print(np.max(xi_values))
    print(np.mean(xi_values))
    batches = len(all_experiments[0])

    title = "Number of Workers: " + str(num_users)
    filename = "rho_comparison_" + dataset + "_" + model + "_" + str(num_users) + "_" + str(local_bs)

    savefile = '../save/' + directory + '/' + filename

    plot_graphs([list(range(batches))] * 2, all_experiments, "", "Epoch", "", ["unidirectional", "bidirectional"], savefile, rotation=rotation)



# def get_xi_values(lrs, num_users=20, epochs=100, model = "mlp", dataset = "mnist", local_bs=10):
#     frac = "1.0"
#     iid = 1
#     topk = 0.001
#     topk_d = 0.001

#     all_experiments = []
#     experiments = []
#     for number in [1]:
#         file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
#                 .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
#                     local_bs, "sparsetopk", lrs[0], 0, topk, topk_d, number)

#         with open(file_name, 'rb') as pickle_file:
#             experiments.append(pickle.load(pickle_file))
#     xi_values = np.mean(np.array(experiments)[:, 4], axis=0)
#     d = len(xi_values) // 100
#     xi_values = batch_to_epoch(xi_values, d)
#     all_experiments.append(xi_values)

#     experiments = []
#     for number in [1]:
#         file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
#                 .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
#                     local_bs, "sparsetopk", lrs[1], 1, topk, topk_d, number)

#         with open(file_name, 'rb') as pickle_file:
#             experiments.append(pickle.load(pickle_file))

#     xi_values = np.mean(np.array(experiments)[:, 4], axis=0)
#     d = len(xi_values) // 100
#     xi_values = batch_to_epoch(xi_values, d)
#     all_experiments.append(xi_values)

#     epochs = len(all_experiments[0])

#     return epochs, all_experiments


def plot_compression_vgg():
    vgg_size = 20040522
    iterations = []
    results = []

    iteration, mnist_10_workers = get_values([0.05, 0.05], num_users=20, epochs=200, model = "vgg", dataset = "cifar", local_bs=100, numbers = [1], index=5)
    iterations.append(list(range(1, iteration + 1)))
    results.append(mnist_10_workers[0]/vgg_size)

    savefile = "compression_vgg"
    savedirectory = "../save/thesis_plots/vgg/"



    plot_graphs_wo_legend([list(range(len(mnist_10_workers[0])))], results, "", "Iteration", "", savedirectory + savefile)



def comparison_without_sgd(num_users, lrs, model="mlp", dataset = "mnist", epochs=100, local_bs=10, index_comparison=3, ylabel="Loss", iid=1, nums=[1]):
    frac = "1.0"
    topk = 0.001
    topk_d = 0.001


    all_experiments = []

    experiments = []
    for num in nums:
        file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", lrs[0], 0, topk, topk_d, num)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file)[index_comparison])

    all_experiments.append(np.average(np.array(experiments), axis=0))

    experiments = []
    for num in nums:
        file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", lrs[1], 1, topk, topk_d, num)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file)[index_comparison])

    all_experiments.append(np.average(np.array(experiments) * 100, axis=0))
    filename = model + "_"+ str(num_users) + "_comparison_" + str(index_comparison)  + "_" + str(iid)

    savefile = '../save/plots/tuning/{}/'.format(dataset) + filename

    plot_graphs([list(range(len(all_experiments[0])))] * len(lrs), all_experiments, "", "Epoch", ylabel, ["unidirectional", "bidirectional", "sgd"], savefile)



def check_file(num_users, lrs, model="mlp", dataset = "mnist", epochs=100, local_bs=10, index_comparison=3, ylabel="Loss", iid=1, nums=[1]):
    frac = "1.0"
    topk = 0.001
    topk_d = 0.001

    all_experiments = []

    experiments = []
    for num in nums:
        file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", lrs[0], 0, topk, topk_d, num)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file)[index_comparison])

    all_experiments.append(np.average(np.array(experiments), axis=0))
    pdb.set_trace()



if __name__ == "__main__": 
    directory = 'check'
    plot_xi_values([0.05, 0.05], num_users=20, epochs=200, model = "vgg", dataset = "cifar", local_bs=100, rotation=0, directory=directory)
    plot_xi_values([0.14, 0.2], num_users=100, epochs=100, model = "cnn", dataset = "fmnist", local_bs=10, rotation=0, directory=directory)
    plot_xi_values([0.13, 0.12], num_users=50, epochs=100, model = "mlp", dataset = "fmnist",  local_bs=10, rotation=0, directory=directory)
    plot_xi_values([0.22, 0.22], num_users=100, epochs=100, model = "mlp", dataset = "fmnist",  local_bs=10, rotation=0, directory=directory)


   
    comparison(20, [0.1, 0.1, 0.05], model="vgg", dataset="cifar", epochs=200, local_bs=100, ylabel="Accuracy", savedirectory= "../save/" + directory)
    comparison(20, [0.1, 0.1, 0.05], model="vgg", dataset="cifar", epochs=200, local_bs=100, ylabel="Loss", index_comparison=0, savedirectory= "../save/" + directory)

    comparison(100, [0.14, 0.20, 0.08], epochs=100, model = "cnn", dataset = "fmnist", index_comparison=0, ylabel="Loss", savedirectory= "../save/" + directory)
    comparison(100, [0.14, 0.20, 0.08], epochs=100, model = "cnn", dataset = "fmnist", index_comparison=3, ylabel="Accuracy", savedirectory= "../save/" + directory)

    comparison(50, [0.13, 0.12, 0.07], epochs=100, model = "mlp", dataset = "fmnist", index_comparison=3, ylabel="Accuracy", savedirectory= "../save/" + directory)
    comparison(50, [0.13, 0.12, 0.07], epochs=100, model = "mlp", dataset = "fmnist", index_comparison=0, ylabel="Loss", savedirectory= "../save/" + directory)

    comparison(100, [0.22, 0.22, 0.08], epochs=100, model = "mlp", dataset = "fmnist", index_comparison=3, ylabel="Accuracy", savedirectory= "../save/" + directory)
    comparison(100, [0.22, 0.22, 0.08], epochs=100, model = "mlp", dataset = "fmnist", index_comparison=0, ylabel="Loss", savedirectory= "../save/" + directory)



    # v, xi = get_downlink([0.14, 0.2], num_users=100, epochs=100, model = "cnn", dataset = "fmnist", numbers=[1], index=4, directory="/Users/williamzou")
    # pdb.set_trace()

