import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_graphs(x, y, title, xlabel, ylabel, legend_labels, savefile):
    plt.figure()
    n = len(x)
    
    for i in range(n):
        plt.plot(x[i], y[i], label=legend_labels[i])
    
    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(savefile  + ".png")



def mnist_tuning(num_users, direction, lrs, model="mlp"):
    dataset = "mnist"
    epochs = 100
    frac = "1.0"
    iid = 1
    local_bs = 10
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



def mnist_sparse_tuning(num_users, direction, lrs, model="mlp"):
    dataset = "mnist"
    epochs = 100
    frac = "1.0"
    iid = 1
    local_bs = 10
    topk = 0.001
    topk_d = 0.001

    all_experiments = []


    for lr in lrs: 
        filename = '{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, num_users, frac, iid, local_bs, "sparsetopk", lr, direction, topk, topk_d, 1)
        
        experiments = []
        
        filepath = '../save/{}-{}/'.format(dataset, model) + filename

        with open(filepath, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))

        all_experiments.append(np.mean(np.array(experiments)[:, 3], axis=0))

    filename = '{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}]' \
                .format(dataset, model, epochs, num_users, frac, iid, local_bs, "sparsetopk", direction, topk, topk_d, 1)

    savefile = '../save/plots/tuning/{}/'.format(dataset) + filename

    plot_graphs([list(range(epochs))] * len(lrs), all_experiments, "", "Epoch", "Accuracy", lrs, savefile)



def mnist_comparison(num_users, lrs, model="mlp"):
    dataset = "mnist"
    epochs = 100
    frac = "1.0"
    iid = 1
    local_bs = 10
    topk = 0.001
    topk_d = 0.001

    all_experiments = []

    experiments = []
    file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
            .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                local_bs, "sparsetopk", lrs[0], 0, topk, topk_d, 1)

    with open(file_name, 'rb') as pickle_file:
        experiments.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments)[:, 3], axis=0))

    experiments = []
    file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
            .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                local_bs, "sparsetopk", lrs[1], 1, topk, topk_d, 1)

    with open(file_name, 'rb') as pickle_file:
        experiments.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments)[:, 3], axis=0))


    experiments = []
    file_name = '../save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
            .format(dataset, model, dataset, model, epochs, num_users, frac, iid,
                local_bs, "sgd", lrs[2], 1, topk, topk_d, 1)

    with open(file_name, 'rb') as pickle_file:
        experiments.append(pickle.load(pickle_file))

    all_experiments.append(np.mean(np.array(experiments)[:, 3], axis=0))
        

    filename = model + "_"+ str(num_users) + "_comparison" 

    savefile = '../save/plots/tuning/{}/'.format(dataset) + filename

    plot_graphs([list(range(epochs))] * len(lrs), all_experiments, "Number of Workers: " + str(num_users), "Epoch", "Accuracy", ["unidirectional (lr = {})".format(lrs[0]), "bidirectional (lr = {})".format(lrs[1]), "sgd (lr = {})".format(lrs[2])], savefile)





def plot_xi_values():
    dataset = "mnist"
    model = "mlp"
    epochs = 100
    num_users = 20
    frac = "1.0"
    iid = 1
    local_bs = 10
    topk = 0.001
    topk_d = 0.001

    all_experiments = []
    experiments = []
    for number in [1]:
        file_name = '../save/mnist-final-v1/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", 0.08, 1, topk, topk_d, number)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))
    xi_values = np.mean(np.array(experiments)[:, 4], axis=0)
    print(xi_values)
    all_experiments.append(xi_values)
    print(np.max(xi_values))
    print(np.mean(xi_values))
    # print(np.mean(experiments))


    experiments = []
    for number in [1]:
        file_name = '../save/mnist-final-v1/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", 0.05, 0, topk, topk_d, number)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))

    xi_values = np.mean(np.array(experiments)[:, 4], axis=0)
    all_experiments.append(xi_values)
    print(np.max(xi_values))
    print(np.mean(xi_values))
    batches = len(all_experiments[0])
    plot_graphs([list(range(batches))] * 2, all_experiments, "", "Batch", r"$\xi$", ["bidirectional", "unidirectional"], "xi_comparison_20")





def plot_xi_values_lhs():
    dataset = "mnist"
    model = "mlp"
    epochs = 30
    num_users = 50
    frac = "1.0"
    iid = 1
    local_bs = 10
    topk = 0.001
    topk_d = 0.001

    all_experiments = []
    experiments = []
    for number in [1]:
        file_name = '../save/mnist-final-v1/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", 0.09, 1, topk, topk_d, number)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))
    xi_values = np.mean(np.array(experiments)[:, 9], axis=0)
    print(xi_values)
    all_experiments.append(xi_values)
    print(np.max(xi_values))
    print(np.mean(xi_values))
    print(np.min(xi_values))
    # print(np.mean(experiments))



    experiments = []
    for number in [1]:
        file_name = '../save/mnist-final-v1/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", 0.08, 0, topk, topk_d, number)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))

    xi_values = np.mean(np.array(experiments)[:, 9], axis=0)
    all_experiments.append(xi_values)
    print(np.max(xi_values))
    print(np.mean(xi_values))
    print(np.min(xi_values))
    batches = len(all_experiments[0])
    plot_graphs([list(range(batches))] * 2, all_experiments, "", "Batch", "Magnitude", ["bidirectional", "unidirectional"], "xi_comparison_20")



def plot_xi_values_rhs():
    dataset = "mnist"
    model = "mlp"
    epochs = 30
    num_users = 10
    frac = "1.0"
    iid = 1
    local_bs = 10
    topk = 0.001
    topk_d = 0.001

    all_experiments = []
    experiments = []
    for number in [1]:
        file_name = '../save/mnist-final-v1/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", 0.05, 1, topk, topk_d, number)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))
    xi_values = np.mean(np.array(experiments)[:, 10], axis=0)
    print(xi_values)
    all_experiments.append(xi_values)
    print(np.max(xi_values))
    print(np.mean(xi_values))
    print(np.min(xi_values))
    # print(np.mean(experiments))



    experiments = []
    for number in [1]:
        file_name = '../save/mnist-final-v1/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl' \
                .format(dataset, model, epochs, num_users, frac, iid,
                    local_bs, "sparsetopk", 0.05, 0, topk, topk_d, number)

        with open(file_name, 'rb') as pickle_file:
            experiments.append(pickle.load(pickle_file))

    xi_values = np.mean(np.array(experiments)[:, 10], axis=0)
    all_experiments.append(xi_values)
    print(np.max(xi_values))
    print(np.mean(xi_values))
    print(np.min(xi_values))
    batches = len(all_experiments[0])
    plot_graphs([list(range(batches))] * 2, all_experiments, "", "Batch", "Magnitude", ["bidirectional", "unidirectional"], "xi_comparison_20")



# 10, 1: 0.05
# 
if __name__ == "__main__":    
    """
    mnist sgd tuning
    """
    # mnist_tuning(50, 1, [0.06, 0.07, 0.08])
    # mnist_tuning(20, 1, [0.01, 0.02, 0.03, 0.04, 0.05])
    # mnist_tuning(100, 1, [0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18])
    # mnist_tuning(10, 1, [0.01, 0.02, 0.03, 0.04, 0.05])



    """
    mnist sparse bidirectional
    """
    mnist_sparse_tuning(10, 1, [0.03, 0.04, 0.05, 0.06, 0.07])
    mnist_sparse_tuning(20, 1, [0.06, 0.07, 0.08, 0.09])
    mnist_sparse_tuning(50, 1, [0.17, 0.18, 0.19, 0.2])
    mnist_sparse_tuning(100, 1, [0.21, 0.22, 0.23, 0.25])


    """
    mnist sparse unidirectional
    """
    # mnist_sparse_tuning(10, 0, [0.04, 0.05, 0.06, 0.07, 0.08])
    # mnist_sparse_tuning(20, 0, [0.07, 0.08, 0.09])
    # mnist_sparse_tuning(50, 0, [0.1, 0.12, 0.15, 0.18])
    # mnist_sparse_tuning(100, 0, [0.1, 0.12, 0.15, 0.18]) 


    """
    mnist comparison
    """
    # mnist_comparison(10, [0.04, 0.07, 0.03])
    mnist_comparison(20, [0.07, 0.09, 0.02])
    # mnist_comparison(50, [0.18, 0.18, 0.08])
    # mnist_comparison(100, [0.15, 0.22, 0.1])

    
    """
    mnist sparse bidirectional
    """   
    mnist_sparse_tuning(10, 1, [0.005, 0.008, 0.01, 0.03, 0.04, 0.05], "cnn")
    mnist_sparse_tuning(20, 1, [0.08, 0.09, 0.1], "cnn")
    mnist_sparse_tuning(50, 1, [0.05, 0.06, 0.08, 0.09, 0.1], "cnn")
    mnist_sparse_tuning(100, 1, [0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14], "cnn")

    
    """
    mnist sparse unidirectional
    """   
    mnist_sparse_tuning(10, 0, [0.06, 0.07, 0.08], "cnn")
    mnist_sparse_tuning(20, 0, [0.01, 0.02, 0.03, 0.05, 0.06, 0.07], "cnn")
    mnist_sparse_tuning(50, 0, [0.05, 0.06, 0.07, 0.08, 0.09], "cnn")
    mnist_sparse_tuning(100, 0, [0.05, 0.06, 0.07, 0.08, 0.09, 0.1], "cnn")



    mnist_tuning(10, 1, [0.01, 0.03, 0.05, 0.07], "cnn")
    mnist_tuning(20, 1, [0.01, 0.03, 0.05, 0.07], "cnn")
    mnist_tuning(50, 1, [0.01, 0.03, 0.05, 0.07], "cnn")
    mnist_tuning(100, 1, [0.01, 0.03, 0.05, 0.07], "cnn")


    mnist_comparison(10, [0.06, 0.05, 0.07], "cnn")
    mnist_comparison(20, [0.07, 0.08, 0.07], "cnn")
    mnist_comparison(50, [0.09, 0.1, 0.07], "cnn")
    mnist_comparison(100, [0.08, 0.14, 0.07], "cnn")

