import math
from typing import List

import cvxopt
import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def define_H(l: float, d: int, m: int):
    H11: np.ndarray = np.identity(d)
    H12: np.ndarray = np.zeros((d, m))
    H21: np.ndarray = np.zeros((m, d))
    H22: np.ndarray = np.zeros((m, m))
    upper_part: np.ndarray = np.concatenate((H11, H12), axis=1)
    lower_part: np.ndarray = np.concatenate((H21, H22), axis=1)
    H: np.ndarray = np.concatenate((upper_part, lower_part), axis=0)
    return 2 * l * H


def define_u(d: int, m: int) -> np.ndarray:
    u1: np.ndarray = np.zeros((d, 1))
    u2: np.ndarray = 1 / m * np.ones((m, 1))
    return np.vstack((u1, u2))


def define_A(d: int, m: int, trainX: np.ndarray, trainY: np.ndarray):
    trainY = trainY.reshape(-1, 1)
    A11: np.ndarray = trainY * trainX
    A12: np.ndarray = np.identity(m)
    A21: np.ndarray = np.zeros((m, d))
    A22: np.ndarray = np.identity(m)
    upper_part: np.ndarray = np.concatenate((A11, A12), axis=1)
    lower_part: np.ndarray = np.concatenate((A21, A22), axis=1)
    A: np.ndarray = np.concatenate((upper_part, lower_part), axis=0)
    return A


def define_v(m: int) -> np.ndarray:
    v1: np.ndarray = np.ones((m, 1))
    v2: np.ndarray = np.zeros((m, 1))
    return np.vstack((v1, v2))


def softsvm(l, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: linear predictor w, a numpy array of size (d, 1)
    """

    # dimensions
    m: int = trainX.shape[0]
    d: int = trainX.shape[1]

    # TODO: consider sparse the matrices
    H: matrix = matrix(define_H(l=l, d=d, m=m))
    u: matrix = matrix(define_u(d=d, m=m))
    A: matrix = matrix(define_A(d=d, m=m, trainX=trainX, trainY=trainy))
    v: matrix = matrix(define_v(m=m))

    sol = solvers.qp(H, u, -A, -v)
    w = np.array(sol["x"])[:d]
    return w


def predictLabels(w: np.ndarray, testX: np.ndarray) -> np.ndarray:
    #w = w.T
    testy_prediction = np.sign(testX @ w)
    testy_prediction[testy_prediction == 0] = 1
    return testy_prediction
    # testy_prediction = np.empty(testX.shape[0], dtype=float)
    # for i, xi in enumerate(testX):
    #     testy_prediction[i] = np.sign(np.inner(w, xi))
    #     if testy_prediction[i] == 0:
    #         testy_prediction[i] = 1
    # return testy_prediction.reshape(-1, 1)


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")


def run_question2_tests():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    train_avg_list: List[float] = []
    train_min_error_list: List[float] = []
    train_max_error_list: List[float] = []

    test_avg_list: List[float] = []
    test_min_error_list: List[float] = []
    test_max_error_list: List[float] = []

    # choice: str = '\0'
    # while choice not in ['1', '2']:
    #     choice = input("Please enter the experiment number you wish to run 1/2: ")

    # -----------------------------------------------Question_2Experiment1----------------------------------------------
    m: int = 100
    iterations: List[int] = list(range(-1, 12))
    iterations.remove(0)
    for n in iterations:
        l: float = 10 ** n
        train_errors: List[float] = []
        test_errors: List[float] = []
        for i in range(1, 11):
            indices = np.random.permutation(trainX.shape[0])
            _trainX = trainX[indices[:m]]
            _trainy = trainy[indices[:m]]

            w: np.ndarray = softsvm(l=l, trainX=_trainX, trainy=_trainy)

            trainy_prediction: np.ndarray = predictLabels(w=w, testX=trainX)
            error: float = np.mean(trainy.flatten() != trainy_prediction.flatten())
            train_errors.append(error)

            testy_prediction: np.ndarray = predictLabels(w=w, testX=testX)
            error: float = np.mean(testy.flatten() != testy_prediction.flatten())
            test_errors.append(error)

            #print(f"l: {l} iteration: {i} error: {error}")

        train_avg: float = np.mean(train_errors)
        train_avg_list.append(round(train_avg, 2))
        train_min_error_list.append(min(train_errors))
        train_max_error_list.append(max(train_errors))

        test_avg: float = np.mean(test_errors)
        test_avg_list.append(round(test_avg, 2))
        test_min_error_list.append(min(test_errors))
        test_max_error_list.append(max(test_errors))

    print("train errors: ", train_avg_list)
    print("test error: ", test_avg_list)

    # TODO: plot the functions for question 2a
    lambdas: List[int] = iterations

    train_lower_error = np.array(train_avg_list) - np.array(train_min_error_list)
    train_upper_error = np.array(train_max_error_list) - np.array(train_avg_list)

    test_lower_error = np.array(test_avg_list) - np.array(test_min_error_list)
    test_upper_error = np.array(test_max_error_list) - np.array(test_avg_list)

    ax = plt.axes()
    ax.errorbar(iterations, train_avg_list, yerr=[train_lower_error, train_upper_error], marker='o',
                label='Train average error', color='green', linestyle='solid', markerfacecolor='red', markersize=8,
                ecolor='darkgreen', capsize=5)
    ax.errorbar(iterations, test_avg_list, yerr=[test_lower_error, test_upper_error], marker='x',
                label='Test average error', color='blue', linestyle='solid', markerfacecolor='orange', markersize=8,
                ecolor='darkred', capsize=5)

    # -----------------------------------------------Question_2Experiment2----------------------------------------------
    m: int = 1000
    train_errors: List[float] = []
    test_errors: List[float] = []
    for n in [1, 3, 5, 8]:
        l: float = 10 ** n
        indices = np.random.permutation(trainX.shape[0])
        _trainX = trainX[indices[:m]]
        _trainy = trainy[indices[:m]]

        w: np.ndarray = softsvm(l=l, trainX=_trainX, trainy=_trainy)

        trainy_prediction: np.ndarray = predictLabels(w=w, testX=trainX)
        error: float = np.mean(trainy.flatten() != trainy_prediction.flatten())
        train_errors.append(error)

        testy_prediction: np.ndarray = predictLabels(w=w, testX=testX)
        error: float = np.mean(testy.flatten() != testy_prediction.flatten())
        test_errors.append(error)

        #print(f"l: {l} error: {error}")

    print("train errors: ", train_errors)
    print("test error: ", test_errors)

    # TODO: plot the functions for question 2b
    lambdas: List[int] = [1, 3, 5, 8]
    ax.scatter(lambdas, train_errors, color='purple', marker='^', s=100, label='Train experiment points')
    ax.scatter(lambdas, test_errors, color='brown', marker='v', s=100, label='Test experiment points')

    # Plot the lines and the points from the 2 experiments
    ax.set(xlim=(-3, 13), ylim=(0, 1), xlabel='lambda', ylabel='error')
    ax.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    #simple_test()

    # here you may add any code that uses the above functions to solve question 2
    run_question2_tests()
