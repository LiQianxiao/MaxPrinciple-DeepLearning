"""
File: plot_logs.py
Project: MaxPrinciple-DeepLearning
File Created: Thursday, 13th December 2018 10:08:04 pm
Author: Qianxiao Li (liqix@ihpc.a-star.edu.sg)
-----
Copyright - 2018 Qianxiao Li, IHPC, A*STAR
License: MIT License
"""

import argparse
import matplotlib.pyplot as plt
import re


def find_float(line):
    """Find and return last floating in line

    Arguments:
        line {string} -- line of text

    Returns:
        float -- last floating number in line
    """

    return float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1])


def chunks(l, n):
    """Yield successive n-sized chunks from l

    Arguments:
        l {list} -- list to split
        n {int} -- chunk size
    """

    for i in range(0, len(l), n):
        yield l[i:i + n]


def process_and_plot(path):
    """Process data and plot training curves

    Arguments:
        path {string} -- log file path
    """

    with open(path, 'r') as file:
        lines = list(file)
        print('Read log file [{}]'.format(path))

    train_losses = [
        find_float(line) for line in lines if 'Train loss:' in line]
    test_losses = [
        find_float(line) for line in lines if 'Test loss:' in line]

    num_all = len(train_losses)
    num_each = num_all // 4

    train_iterator = chunks(train_losses, num_each)
    train_losses_msa = next(train_iterator)
    train_losses_sgd = next(train_iterator)
    train_losses_adagrad = next(train_iterator)
    train_losses_adam = next(train_iterator)

    test_iterator = chunks(test_losses, num_each)
    test_losses_msa = next(test_iterator)
    test_losses_sgd = next(test_iterator)
    test_losses_adagrad = next(test_iterator)
    test_losses_adam = next(test_iterator)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.semilogy(train_losses_msa, '-', label='MSA')
    ax1.semilogy(train_losses_sgd, ':', label='SGD')
    ax1.semilogy(train_losses_adagrad, '--', label='Adagrad')
    ax1.semilogy(train_losses_adam, '-.', label='Adam')
    ax1.set_title('Test losses')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.semilogy(test_losses_msa, '-', label='MSA')
    ax2.semilogy(test_losses_sgd, ':', label='SGD')
    ax2.semilogy(test_losses_adagrad, '--', label='Adagrad')
    ax2.semilogy(test_losses_adam, '-.', label='Adam')
    ax2.set_title('Train losses')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot loss from log files')
    parser.add_argument(
        '--logdir',
        help='directory to log file: python plot_logs.py --path {name}.log')
    args = parser.parse_args()

    process_and_plot(args.logdir)
