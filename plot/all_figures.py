import pdb
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from retrieval import get_xi_values, get_side_values, get_side_values_batches, get_model_results, get_values, get_distance, batch_to_epoch_avg


save = "check"

def compare_side_by_side(x, files, savefile):
    fig = plt.figure(constrained_layout=True, figsize=(5,7))
    fig.supylabel('Magnitude')
    fig.supxlabel('Epoch')
    worker = [20, 50, 100]
    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=3, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f'{worker[row]} Workers')

        subset = files[row]

        r = [[subset[0][0], subset[1][0]], [subset[0][1], subset[1][1]]]
        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=2, sharey=True)
        for col, ax in enumerate(axs):
            ax.plot(x, r[col][0], label="unidirectional")
            ax.plot(x, r[col][1], label="bidirectional")
            ax.grid()
            if col == 1 and row == 0:
                ax.legend()


            handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center')
    savedirectory = "../save/" + save + "/"
    plt.savefig(savedirectory + savefile + ".pdf")

def plot_subplot(ax, x, y, title, legend_labels):
    n = len(y)
    
    for i in range(n):
        ax.plot(x, y[i], label=legend_labels[i], linewidth=1)
    
    ax.grid()
    ax.set_title(title)


def plot_graphs_wo_legend(x, y, title, xlabel, ylabel, savefile, rotation=90):
    plt.figure(figsize=(4, 3))
    n = len(x)
    
    for i in range(n):
        plt.plot(x[i], y[i], linewidth=1)
       
    

    plt.xlabel(xlabel)
    plt.ylabel(ylabel, labelpad=10, rotation=rotation)
    plt.ylim(0, 0.02)
    plt.grid()
    plt.tight_layout() 
    # plt.show()
    plt.savefig(savefile  + ".pdf",  bbox_inches='tight')


def plot_compression_vgg():
    vgg_size = 20040522
    iterations = []
    results = []

    iteration, mnist_20_workers = get_values([0.05, 0.05], num_users=20, epochs=200, model = "vgg", dataset = "cifar", local_bs=100, numbers = [1], index=5)
    avg_20 = batch_to_epoch_avg(mnist_20_workers[0], 25) 
    # pdb.set_trace()
    results.append(np.array(avg_20)/vgg_size)


    savefile = "compression_vgg"

    savedirectory = "../save/" + save + "/vgg/"


    plot_graphs_wo_legend([list(range(1, 201))], results, "", "Epoch", "", savedirectory + savefile, rotation=0)



def comparison_between_workers_without_comparison(x, y, xlabel, ylabel, savefile, rotation=0, sharey=False):
    fig, axs = plt.subplots(1, 3, figsize=(8, 2.5), sharey=sharey)
    # pdb.set_trace()

    axs[0].plot(x, y[0], linewidth=1)
    axs[0].set_title("20 Workers")
    # axs[0].set_ylim([0, 0.021])
    axs[0].grid()
    # pdb.set_trace()
    axs[1].plot(x, y[1], linewidth=1)
    axs[1].grid()
    # axs[1].set_ylim([0, 0.05])
    axs[1].set_title("50 Workers")
    axs[2].plot(x, y[2], linewidth=1)
    axs[2].set_title("100 Workers")
    # axs[2].set_ylim([0, 0.1])
    axs[2].grid()

    fig.text(0.51, 0.01, xlabel, ha='center')

    fig.text(-0.01, 0.5, ylabel, va='center', rotation=rotation)
    fig.tight_layout() 

    savedirectory = "../save/" + save + "/"
    # plt.show()
    plt.savefig(savedirectory + savefile + ".pdf", bbox_inches='tight')


def comparison_between_workers(x, y1, y2, y3, ylabel, savefile, sgd=0, rotation=0, sharey=False, middle=0.75):
    fig, axs = plt.subplots(1, 3, figsize=(8, 2.5), sharey=sharey)
    
    if sgd == 0:
        legend = ["unidirectional", "bidirectional"]
    else:
        legend = ["unidirectional", "bidirectional", "sgd"]

    plot_subplot(axs[0], x, y1, "20 workers", legend)
    plot_subplot(axs[1], x, y2, "50 workers", legend)
    plot_subplot(axs[2], x, y3, "100 workers", legend)

    fig.text(0.51, 0.01, 'Epoch', ha='center')

    fig.text(0.04, 0.5, ylabel, va='center', rotation=rotation)
    handles, labels = axs[2].get_legend_handles_labels()

    fig.subplots_adjust(bottom=0.2)
    fig.legend(handles, labels, bbox_to_anchor=(middle, 0.001), ncol=3)

    savedirectory = "../save/" + save + "/"

    plt.savefig(savedirectory + savefile + ".pdf", bbox_inches='tight')


def plot_test_results():
    epochs, fmnist_20_workers = get_model_results([0.08, 0.08, 0.06], num_users=20, epochs=100, model = "mlp", dataset = "fmnist", numbers=[1])
    _, fmnist_50_workers = get_model_results([0.13, 0.12, 0.07], num_users=50, epochs=100, model = "mlp", dataset = "fmnist", numbers=[1])
    _, fmnist_100_workers = get_model_results([0.22, 0.22, 0.08], num_users=100, epochs=100, model = "mlp", dataset = "fmnist", numbers=[1])

    xlabel = list(range(1, epochs + 1))
    comparison_between_workers(xlabel, fmnist_20_workers, fmnist_50_workers, fmnist_100_workers, "Accuracy", "fmnist_mlp_test_accuracy", sgd=1, rotation=90, sharey=True)


    _, fmnist_20_workers = get_model_results([0.08, 0.08, 0.06], num_users=20, epochs=100, model = "mlp", dataset = "fmnist", numbers=[1], index_comparison=0)
    _, fmnist_50_workers = get_model_results([0.13, 0.12, 0.07], num_users=50, epochs=100, model = "mlp", dataset = "fmnist", numbers=[1], index_comparison=0)
    _, fmnist_100_workers = get_model_results([0.22, 0.22, 0.08], num_users=100, epochs=100, model = "mlp", dataset = "fmnist", numbers=[1], index_comparison=0)


    xlabel = list(range(1, epochs + 1))
    comparison_between_workers(xlabel, fmnist_20_workers, fmnist_50_workers, fmnist_100_workers, "Loss", "fmnist_mlp_train_loss", sgd=1, rotation=90, sharey=True)



    _, fmnist_20_workers = get_model_results([0.09, 0.08, 0.06], num_users=20, epochs=100, model = "cnn", dataset = "fmnist", numbers=[1])
    _, fmnist_50_workers = get_model_results([0.12, 0.11, 0.07], num_users=50, epochs=100, model = "cnn", dataset = "fmnist", numbers=[1])
    _, fmnist_100_workers = get_model_results([0.14, 0.20, 0.08], num_users=100, epochs=100, model = "cnn", dataset = "fmnist", numbers=[1])

    xlabel = list(range(1, epochs + 1))
    comparison_between_workers(xlabel, fmnist_20_workers, fmnist_50_workers, fmnist_100_workers, "Accuracy", "fmnist_cnn_test_accuracy", sgd=1, rotation=90, sharey=True)


    _, fmnist_20_workers = get_model_results([0.09, 0.08, 0.06], num_users=20, epochs=100, model = "cnn", dataset = "fmnist", numbers=[1], index_comparison=0)
    _, fmnist_50_workers = get_model_results([0.12, 0.11, 0.07], num_users=50, epochs=100, model = "cnn", dataset = "fmnist", numbers=[1], index_comparison=0)
    _, fmnist_100_workers = get_model_results([0.14, 0.20, 0.08], num_users=100, epochs=100, model = "cnn", dataset = "fmnist", numbers=[1], index_comparison=0)

    xlabel = list(range(1, epochs + 1))
    comparison_between_workers(xlabel, fmnist_20_workers, fmnist_50_workers, fmnist_100_workers, "Loss", "fmnist_cnn_train_loss", sgd=1, rotation=90, sharey=True)





    # epochs, mnist_10_workers = get_model_results([0.04, 0.1, 0.03], num_users=10, epochs=100, model = "mlp", dataset = "mnist", numbers=[1], index_comparison=0)
    _, mnist_20_workers = get_model_results([0.06, 0.09, 0.03], num_users=20, epochs=100, model = "mlp", dataset = "mnist", numbers=[1], index_comparison=0)
    _, mnist_50_workers = get_model_results([0.17, 0.18, 0.1], num_users=50, epochs=100, model = "mlp", dataset = "mnist", numbers=[1], index_comparison=0)
    _, mnist_100_workers = get_model_results([0.17, 0.24, 0.1], num_users=100, epochs=100, model = "mlp", dataset = "mnist", numbers=[1], index_comparison=0)

    xlabel = list(range(1, epochs + 1))
    comparison_between_workers(xlabel, mnist_20_workers, fmnist_50_workers, mnist_100_workers, "Loss", "mnist_mlp_train_loss", sgd=1, rotation=90, sharey=True)


    epochs, mnist_10_workers = get_model_results([0.04, 0.1, 0.03], num_users=10, epochs=100, model = "mlp", dataset = "mnist", numbers=[1], index_comparison=3)
    _, mnist_20_workers = get_model_results([0.06, 0.09, 0.03], num_users=20, epochs=100, model = "mlp", dataset = "mnist", numbers=[1], index_comparison=3)
    _, mnist_50_workers = get_model_results([0.17, 0.18, 0.1], num_users=50, epochs=100, model = "mlp", dataset = "mnist", numbers=[1], index_comparison=3)
    _, mnist_100_workers = get_model_results([0.17, 0.24, 0.1], num_users=100, epochs=100, model = "mlp", dataset = "mnist", numbers=[1], index_comparison=3)

    xlabel = list(range(1, epochs + 1))
    comparison_between_workers(xlabel, mnist_20_workers, mnist_50_workers, mnist_100_workers, "Accuracy", "mnist_mlp_test_accuracy", sgd=1, rotation=90, sharey=True)


 

    epochs, mnist_10_workers = get_model_results([0.06, 0.07, 0.05], num_users=10, epochs=100, model = "cnn", dataset = "mnist", numbers=[1], index_comparison=0, sgddir=1)
    _, mnist_20_workers = get_model_results([0.08, 0.09, 0.05], num_users=20, epochs=100, model = "cnn", dataset = "mnist", numbers=[1], index_comparison=0, sgddir=1)
    _, mnist_50_workers = get_model_results([0.14, 0.16, 0.07], num_users=50, epochs=100, model = "cnn", dataset = "mnist", numbers=[1], index_comparison=0, sgddir=1)
    _, mnist_100_workers = get_model_results([0.09, 0.16, 0.07], num_users=100, epochs=100, model = "cnn", dataset = "mnist", numbers=[1], index_comparison=0, sgddir=1)

    xlabel = list(range(1, epochs + 1))
    comparison_between_workers(xlabel, mnist_20_workers, mnist_50_workers, mnist_100_workers, "Loss", "mnist_cnn_train_loss", sgd=1, rotation=90, sharey=True)


    epochs, mnist_10_workers = get_model_results([0.06, 0.07, 0.05], num_users=10, epochs=100, model = "cnn", dataset = "mnist", numbers=[1], index_comparison=3, sgddir=1)
    _, mnist_20_workers = get_model_results([0.08, 0.09, 0.05], num_users=20, epochs=100, model = "cnn", dataset = "mnist", numbers=[1], index_comparison=3, sgddir=1)
    _, mnist_50_workers = get_model_results([0.14, 0.16, 0.07], num_users=50, epochs=100, model = "cnn", dataset = "mnist", numbers=[1], index_comparison=3, sgddir=1)
    _, mnist_100_workers = get_model_results([0.09, 0.16, 0.07], num_users=100, epochs=100, model = "cnn", dataset = "mnist", numbers=[1], index_comparison=3, sgddir=1)

    xlabel = list(range(1, epochs + 1))
    comparison_between_workers(xlabel, mnist_20_workers, mnist_50_workers, mnist_100_workers, "Accuracy", "mnist_cnn_test_accuracy", sgd=1, rotation=90, sharey=True)


def plot_compression():
    mnist_mlp_size = 50890
    iterations = []
    results = []

    iteration, mnist_20_workers = get_values([0.06, 0.09], num_users=20, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [2], index=5)
    avg_20 = batch_to_epoch_avg(mnist_20_workers[0], 300) 
    results.append(np.array(avg_20)/mnist_mlp_size)

    iteration, mnist_50_workers = get_values([0.17, 0.18], num_users=50, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [2], index=5)
    avg_50 = batch_to_epoch_avg(mnist_50_workers[0], 120) 
    results.append(np.array(avg_50)/mnist_mlp_size)

    iteration, mnist_100_workers = get_values([0.17, 0.24], num_users=100, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [2], index=5)
    avg_100 = batch_to_epoch_avg(mnist_100_workers[0], 60)
    results.append(np.array(avg_100)/mnist_mlp_size)
    comparison_between_workers_without_comparison(list(range(1, 101)), results, "Epoch", "", "mnist_mlp_compression_upstream", rotation=0)

    
    mnist_mlp_size = 1199882
    results = []

    iteration, mnist_20_workers = get_values([0.08, 0.09], num_users=20, epochs=100, model = "cnn", dataset = "mnist", local_bs=10, numbers = [2], index=5)
    avg_20 = batch_to_epoch_avg(mnist_20_workers[0], 300) 
    results.append(np.array(avg_20)/mnist_mlp_size)


    iteration, mnist_50_workers = get_values([0.14, 0.16], num_users=50, epochs=100, model = "cnn", dataset = "mnist", local_bs=10, numbers = [2], index=5)
    avg_50 = batch_to_epoch_avg(mnist_50_workers[0], 120) 
    results.append(np.array(avg_50)/mnist_mlp_size)

    iteration, mnist_100_workers = get_values([0.09, 0.16], num_users=100, epochs=100, model = "cnn", dataset = "mnist", local_bs=10, numbers = [2], index=5)
    avg_100 = batch_to_epoch_avg(mnist_100_workers[0], 60)
    results.append(np.array(avg_100)/mnist_mlp_size)
    # pdb.set_trace()
    comparison_between_workers_without_comparison(list(range(1, 101)), results, "Epoch", "", "mnist_cnn_compression_upstream", rotation=0)


    mnist_mlp_size = 242762
    iterations = []
    results = []

    iteration, mnist_20_workers = get_values([0.08, 0.08], num_users=20, epochs=100, model = "mlp", dataset = "fmnist", local_bs=10, numbers = [2], index=5)
    avg_20 = batch_to_epoch_avg(mnist_20_workers[0], 300) 
    results.append(np.array(avg_20)/mnist_mlp_size)

    iteration, mnist_50_workers = get_values([0.13, 0.12], num_users=50, epochs=100, model = "mlp", dataset = "fmnist", local_bs=10, numbers = [2], index=5)
    avg_50 = batch_to_epoch_avg(mnist_50_workers[0], 120) 
    results.append(np.array(avg_50)/mnist_mlp_size)

    iteration, mnist_100_workers = get_values([0.22, 0.22], num_users=100, epochs=100, model = "mlp", dataset = "fmnist", local_bs=10, numbers = [2], index=5)
    avg_100 = batch_to_epoch_avg(mnist_100_workers[0], 60)
    results.append(np.array(avg_100)/mnist_mlp_size)

    comparison_between_workers_without_comparison(list(range(1, 101)), results, "Epoch", "", "fmnist_mlp_compression_upstream", rotation=0)



    mnist_mlp_size = 29034
    results = []

    iteration, mnist_20_workers = get_values([0.09, 0.08], num_users=20, epochs=100, model = "cnn", dataset = "fmnist", local_bs=10, numbers = [1], index=5)
    avg_20 = batch_to_epoch_avg(mnist_20_workers[0], 300) 
    results.append(np.array(avg_20)/mnist_mlp_size)

    iteration, mnist_50_workers = get_values([0.12, 0.11], num_users=50, epochs=100, model = "cnn", dataset = "fmnist", local_bs=10, numbers = [1], index=5)
    avg_50 = batch_to_epoch_avg(mnist_50_workers[0], 120) 
    results.append(np.array(avg_50)/mnist_mlp_size)

    iteration, mnist_100_workers = get_values([0.14, 0.2], num_users=100, epochs=100, model = "cnn", dataset = "fmnist", local_bs=10, numbers = [1], index=5)
    avg_100 = batch_to_epoch_avg(mnist_100_workers[0], 60)
    results.append(np.array(avg_100)/mnist_mlp_size)
    comparison_between_workers_without_comparison(list(range(1, 101)), results, "Epoch", "", "fmnist_cnn_compression_upstream", rotation=0)


def plot_rho():
    epochs, fmnist_20_workers = get_xi_values([0.08, 0.08], num_users=20, epochs=100, model = "mlp", dataset = "fmnist",  numbers=[2])   
    _, fmnist_50_workers = get_xi_values([0.13, 0.12], num_users=50, epochs=100, model = "mlp", dataset = "fmnist",  numbers=[2])   
    _, fmnist_100_workers = get_xi_values([0.22, 0.22], num_users=100, epochs=100, model = "mlp", dataset = "fmnist",  numbers=[2])

    xlabel = list(range(1, epochs + 1))
    comparison_between_workers(xlabel, fmnist_20_workers, fmnist_50_workers, fmnist_100_workers, "", "fmnist_mlp_rho", rotation=0, middle=0.7)



    epochs, fmnist_20_workers = get_xi_values([0.09, 0.08], num_users=20, epochs=100, model = "cnn", dataset = "fmnist", numbers=[1])
    _, fmnist_50_workers = get_xi_values([0.12, 0.11], num_users=50, epochs=100, model = "cnn", dataset = "fmnist", numbers=[1])
    _, fmnist_100_workers = get_xi_values([0.14, 0.2], num_users=100, epochs=100, model = "cnn", dataset = "fmnist", numbers=[1])

    xlabel = list(range(1, epochs + 1))
    comparison_between_workers(xlabel, fmnist_20_workers, fmnist_50_workers, fmnist_100_workers, "", "fmnist_cnn_rho", rotation=0, middle=0.7)




    epochs, mnist_20_workers = get_xi_values([0.06, 0.09], num_users=20, epochs=100, model = "mlp", dataset = "mnist", numbers=[2])
    _, mnist_50_workers = get_xi_values([0.17, 0.18], num_users=50, epochs=100, model = "mlp", dataset = "mnist", numbers=[2])
    _, mnist_100_workers = get_xi_values([0.17, 0.24], num_users=100, epochs=100, model = "mlp", dataset = "mnist", numbers=[2])

    xlabel = list(range(1, epochs + 1))
    comparison_between_workers(xlabel, mnist_20_workers, mnist_50_workers, mnist_100_workers, "", "mnist_mlp_rho", rotation=0, middle=0.7)


    _, mnist_20_workers = get_xi_values([0.08, 0.09], num_users=20, epochs=100, model = "cnn", dataset = "mnist", numbers=[2])
    _, mnist_50_workers = get_xi_values([0.14, 0.16], num_users=50, epochs=100, model = "cnn", dataset = "mnist", numbers=[2])
    _, mnist_100_workers = get_xi_values([0.09, 0.16], num_users=100, epochs=100, model = "cnn", dataset = "mnist", numbers=[2])


    xlabel = list(range(1, epochs + 1))
    comparison_between_workers(xlabel, mnist_20_workers, mnist_50_workers, mnist_100_workers, "", "mnist_cnn_rho")



def plot_rho_sides():
    files = []

    epochs, fmnist_20_workers_0 = get_side_values(0.08, num_users=20, epochs=100, model = "mlp", dataset = "fmnist", local_bs=10, numbers = [1], direction=0)
    epochs, fmnist_20_workers_1 = get_side_values(0.08, num_users=20, epochs=100, model = "mlp", dataset = "fmnist", local_bs=10, numbers = [1], direction=1)

    files.append([fmnist_20_workers_0, fmnist_20_workers_1])

    epochs, fmnist_50_workers_0 = get_side_values(0.13, num_users=50, epochs=100, model = "mlp", dataset = "fmnist", local_bs=10, numbers = [1], direction=0)
    epochs, fmnist_50_workers_1 = get_side_values(0.12, num_users=50, epochs=100, model = "mlp", dataset = "fmnist", local_bs=10, numbers = [1], direction=1)

    files.append([fmnist_50_workers_0, fmnist_50_workers_1])

    epochs, fmnist_100_workers_0 = get_side_values(0.22, num_users=100, epochs=100, model = "mlp", dataset = "fmnist", local_bs=10, numbers = [1], direction=0)
    epochs, fmnist_100_workers_1 = get_side_values(0.22, num_users=100, epochs=100, model = "mlp", dataset = "fmnist", local_bs=10, numbers = [1], direction=1)

    files.append([fmnist_100_workers_0, fmnist_100_workers_1])
    xlabel = list(range(1, epochs + 1))

    compare_side_by_side(xlabel, files, "fmnist_mlp_side_by_side")



    files = []

    epochs, fmnist_20_workers_0 = get_side_values(0.09, num_users=20, epochs=100, model = "cnn", dataset = "fmnist", local_bs=10, numbers = [1], direction=0)
    epochs, fmnist_20_workers_1 = get_side_values(0.08, num_users=20, epochs=100, model = "cnn", dataset = "fmnist", local_bs=10, numbers = [1], direction=1)

    files.append([fmnist_20_workers_0, fmnist_20_workers_1])

    epochs, fmnist_50_workers_0 = get_side_values(0.12, num_users=50, epochs=100, model = "cnn", dataset = "fmnist", local_bs=10, numbers = [1], direction=0)
    epochs, fmnist_50_workers_1 = get_side_values(0.11, num_users=50, epochs=100, model = "cnn", dataset = "fmnist", local_bs=10, numbers = [1], direction=1)

    files.append([fmnist_50_workers_0, fmnist_50_workers_1])

    epochs, fmnist_100_workers_0 = get_side_values(0.14, num_users=100, epochs=100, model = "cnn", dataset = "fmnist", local_bs=10, numbers = [1], direction=0)
    epochs, fmnist_100_workers_1 = get_side_values(0.2, num_users=100, epochs=100, model = "cnn", dataset = "fmnist", local_bs=10, numbers = [1], direction=1)

    files.append([fmnist_100_workers_0, fmnist_100_workers_1])
    xlabel = list(range(1, epochs + 1))
    compare_side_by_side(xlabel, files, "fmnist_cnn_side_by_side")

    files = []

    epochs, mnist_20_workers_0 = get_side_values(0.08, num_users=20, epochs=100, model = "cnn", dataset = "mnist", local_bs=10, numbers = [2], direction=0)
    epochs, mnist_20_workers_1 = get_side_values(0.09, num_users=20, epochs=100, model = "cnn", dataset = "mnist", local_bs=10, numbers = [2], direction=1)

    files.append([mnist_20_workers_0, mnist_20_workers_1])

    epochs, mnist_50_workers_0 = get_side_values(0.14, num_users=50, epochs=100, model = "cnn", dataset = "mnist", local_bs=10, numbers = [2], direction=0)
    epochs, mnist_50_workers_1 = get_side_values(0.16, num_users=50, epochs=100, model = "cnn", dataset = "mnist", local_bs=10, numbers = [2], direction=1)

    files.append([mnist_50_workers_0, mnist_50_workers_1])

    epochs, mnist_100_workers_0 = get_side_values(0.09, num_users=100, epochs=100, model = "cnn", dataset = "mnist", local_bs=10, numbers = [2], direction=0)
    epochs, mnist_100_workers_1 = get_side_values(0.16, num_users=100, epochs=100, model = "cnn", dataset = "mnist", local_bs=10, numbers = [2], direction=1)

    files.append([mnist_100_workers_0, mnist_100_workers_1])
    xlabel = list(range(1, epochs + 1))
    compare_side_by_side(xlabel, files, "mnist_cnn_side_by_side")

    files = []

    epochs, mnist_20_workers_0 = get_side_values(0.06, num_users=20, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [2], direction=0)
    epochs, mnist_20_workers_1 = get_side_values(0.09, num_users=20, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [2], direction=1)

    files.append([mnist_20_workers_0, mnist_20_workers_1])

    epochs, mnist_50_workers_0 = get_side_values(0.17, num_users=50, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [2], direction=0)
    epochs, mnist_50_workers_1 = get_side_values(0.18, num_users=50, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [2], direction=1)

    files.append([mnist_50_workers_0, mnist_50_workers_1])

    epochs, mnist_100_workers_0 = get_side_values(0.17, num_users=100, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [2], direction=0)
    epochs, mnist_100_workers_1 = get_side_values(0.24, num_users=100, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [2], direction=1)

    files.append([mnist_100_workers_0, mnist_100_workers_1])
    xlabel = list(range(1, epochs + 1))
    compare_side_by_side(xlabel, files, "mnist_mlp_side_by_side")



def compare_delta_values(x, files, savefile):
    fig = plt.figure(constrained_layout=True, figsize=(5,8))
    # fig.suptitle(figure_title)
    fig.supylabel(r"$1 - \gamma$", rotation=90)
    fig.supxlabel('Epoch')
    worker = [20, 50, 100]
    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=3, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f'{worker[row]} Workers')

        subset = files[row]

        
        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=2, sharey=True)
        for col, ax in enumerate(axs):
            if col == 0:
                ax.plot(x, subset[0])
            else:
                ax.plot(x, subset[1], label="uplink")
                ax.plot(x, subset[2], label="downlink")
            ax.grid()
            if col == 0 and row == 0:
                ax.set_title("Unidirectional")
            elif col == 1 and row == 0:
                ax.set_title("Bidirectional")
            if col == 1 and row == 0:
                ax.legend()


            handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center')
    savedirectory = "../save/" + save + "/"
    # plt.show()
    plt.savefig(savedirectory + savefile + ".pdf")


def plot_gamma_values():
    files = []

    epochs, fmnist_20_workers_u = get_xi_values([0.08, 0.08], num_users=20, epochs=100, model = "mlp", dataset = "fmnist", local_bs=10, numbers = [1], index=7)
    epochs, fmnist_20_workers_d = get_xi_values([0.08, 0.08], num_users=20, epochs=100, model = "mlp", dataset = "fmnist", local_bs=10, numbers = [1], index=8)
    files.append([fmnist_20_workers_u[0][1:100], fmnist_20_workers_u[1][1:100], fmnist_20_workers_d[1][1:100]])


    epochs, fmnist_50_workers_u = get_xi_values([0.13, 0.12], num_users=50, epochs=100, model = "mlp", dataset = "fmnist", local_bs=10, numbers = [1], index=7)
    epochs, fmnist_50_workers_d = get_xi_values([0.13, 0.12], num_users=50, epochs=100, model = "mlp", dataset = "fmnist", local_bs=10, numbers = [1], index=8)
    files.append([fmnist_50_workers_u[0][1:100], fmnist_50_workers_u[1][1:100], fmnist_50_workers_d[1][1:100]])


    epochs, fmnist_100_workers_u = get_xi_values([0.22, 0.22], num_users=100, epochs=100, model = "mlp", dataset = "fmnist", local_bs=10, numbers = [1], index=7)
    epochs, fmnist_100_workers_d = get_xi_values([0.22, 0.22], num_users=100, epochs=100, model = "mlp", dataset = "fmnist", local_bs=10, numbers = [1], index=8)
    files.append([fmnist_100_workers_u[0][1:100], fmnist_100_workers_u[1][1:100], fmnist_100_workers_d[1][1:100]])


    xlabel = list(range(1, 100))
    compare_delta_values(xlabel, files, "fmnist_mlp_delta_values")


    files = []

    epochs, fmnist_20_workers_u = get_xi_values([0.06, 0.09], num_users=20, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [2], index=7)
    epochs, fmnist_20_workers_d = get_xi_values([0.06, 0.09], num_users=20, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [2], index=8)
    files.append([fmnist_20_workers_u[0][1:100], fmnist_20_workers_u[1][1:100], fmnist_20_workers_d[1][1:100]])


    epochs, fmnist_50_workers_u = get_xi_values([0.17, 0.18], num_users=50, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [2], index=7)
    epochs, fmnist_50_workers_d = get_xi_values([0.17, 0.18], num_users=50, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [2], index=8)
    files.append([fmnist_50_workers_u[0][1:100], fmnist_50_workers_u[1][1:100], fmnist_50_workers_d[1][1:100]])


    epochs, fmnist_100_workers_u = get_xi_values([0.17, 0.24], num_users=100, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [2], index=7)
    epochs, fmnist_100_workers_d = get_xi_values([0.17, 0.24], num_users=100, epochs=100, model = "mlp", dataset = "mnist", local_bs=10, numbers = [2], index=8)
    files.append([fmnist_100_workers_u[0][1:100], fmnist_100_workers_u[1][1:100], fmnist_100_workers_d[1][1:100]])
    xlabel = list(range(1, 100))
    compare_delta_values(xlabel, files, "mnist_mlp_delta_values")


    files = []
    epochs, fmnist_20_workers_u = get_xi_values([0.08, 0.09], num_users=20, epochs=100, model = "cnn", dataset = "mnist", local_bs=10, numbers = [2], index=7)
    epochs, fmnist_20_workers_d = get_xi_values([0.08, 0.09], num_users=20, epochs=100, model = "cnn", dataset = "mnist", local_bs=10, numbers = [2], index=8)
    files.append([fmnist_20_workers_u[0][1:100], fmnist_20_workers_u[1][1:100], fmnist_20_workers_d[1][1:100]])


    epochs, fmnist_50_workers_u = get_xi_values([0.14, 0.16], num_users=50, epochs=100, model = "cnn", dataset = "mnist", local_bs=10, numbers = [2], index=7)
    epochs, fmnist_50_workers_d = get_xi_values([0.14, 0.16], num_users=50, epochs=100, model = "cnn", dataset = "mnist", local_bs=10, numbers = [2], index=8)
    files.append([fmnist_50_workers_u[0][1:100], fmnist_50_workers_u[1][1:100], fmnist_50_workers_d[1][1:100]])


    epochs, fmnist_100_workers_u = get_xi_values([0.09, 0.16], num_users=100, epochs=100, model = "cnn", dataset = "mnist", local_bs=10, numbers = [2], index=7)
    epochs, fmnist_100_workers_d = get_xi_values([0.09, 0.16], num_users=100, epochs=100, model = "cnn", dataset = "mnist", local_bs=10, numbers = [2], index=8)
    files.append([fmnist_100_workers_u[0][1:100], fmnist_100_workers_u[1][1:100], fmnist_100_workers_d[1][1:100]])
    xlabel = list(range(1, 100 ))
    compare_delta_values(xlabel, files, "mnist_cnn_delta_values")


    files = []
    epochs, fmnist_20_workers_u = get_xi_values([0.09, 0.08], num_users=20, epochs=100, model = "cnn", dataset = "fmnist", local_bs=10, numbers = [1], index=7)
    epochs, fmnist_20_workers_d = get_xi_values([0.09, 0.08], num_users=20, epochs=100, model = "cnn", dataset = "fmnist", local_bs=10, numbers = [1], index=8)
    files.append([fmnist_20_workers_u[0][1:100], fmnist_20_workers_u[1][1:100], fmnist_20_workers_d[1][1:100]])


    epochs, fmnist_50_workers_u = get_xi_values([0.12, 0.11], num_users=50, epochs=100, model = "cnn", dataset = "fmnist", local_bs=10, numbers = [1], index=7)
    epochs, fmnist_50_workers_d = get_xi_values([0.12, 0.11], num_users=50, epochs=100, model = "cnn", dataset = "fmnist", local_bs=10, numbers = [1], index=8)
    files.append([fmnist_50_workers_u[0][1:100], fmnist_50_workers_u[1][1:100], fmnist_50_workers_d[1][1:100]])


    epochs, fmnist_100_workers_u = get_xi_values([0.14, 0.2], num_users=100, epochs=100, model = "cnn", dataset = "fmnist", local_bs=10, numbers = [1], index=7)
    epochs, fmnist_100_workers_d = get_xi_values([0.14, 0.2], num_users=100, epochs=100, model = "cnn", dataset = "fmnist", local_bs=10, numbers = [1], index=8)
    files.append([fmnist_100_workers_u[0][1:100], fmnist_100_workers_u[1][1:100], fmnist_100_workers_d[1][1:100]])
    xlabel = list(range(1, 100))
    compare_delta_values(xlabel, files, "fmnist_cnn_delta_values")


if __name__ ==  "__main__":
    plot_gamma_values()
    plot_test_results()

    plot_compression()
    plot_compression_vgg()

    plot_rho_sides()
    plot_rho()




