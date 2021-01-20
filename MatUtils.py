"""This a math utility file required for training the network"""
import math

"""This function does the transpose of a Matrix
[Parameters] : Input list which has to be transposed
Return value:
[result]     : Returns transposed list"""
def transpose(lst):
    if isinstance(lst[0], list):
        r = [list(x) for x in zip(*lst)]
    else:
        r = [list(x) for x in zip(lst)]
    return r

"""This function does the matrix multiplication of two Matrix
[list_1] : Input list 1
[list_2] : Input list 2
Return value:
[result]     : Returns matrix dot product"""
def dot(X, Y):
    result = []  # final result
    col = 0
    for row in Y:
        for a, b in zip(X, row):
            if isinstance(a, list):
                a = a[0]
            col = col + a * b
        result.append(col)
        col = 0
    ret = result[:]
    result.clear()
    return ret


"""This function does the addition of two Matrix
[X] : Input list 1
[Y] : Input list 2
Return value:
[result]     : Returns matrix after adding each element"""
def add(X, Y):
    if isinstance(X[0], list):
        result = [[c + d for c, d in zip(a, b)] for a, b in zip(X, Y)]
    else:
        result = [a + b for a, b in zip(X, Y)]
    return result

'''This is an activation function used for scaling the output between 0 & 1
Mathematical formula used: 1 / 1+ exp**(-lamda*node_value)
Parameters:
[obj] : object of Neurons class
[node_value]   : This can be either input neuron or hidden neuron value.
Return Value:
[ret]           : Normalized value of node_value'''
def sigmoid(obj, x):
    ret = [math.exp(-number * obj._lamda) for number in x]
    ret = [1 + number for number in ret]
    ret = [1 / number for number in ret]
    return ret

'''This is function calculates the derivative.
Mathematical formula used: lambda * node_value * (1 - node_value)
Parameters:
[obj] : object of Neurons class
[node_value]   : This can be either hidden neuron or output neuron value.
Return Value:
[x3]           : Drivative Output'''
def sigmoid_derivative(obj, x):
    x1 = [1 - number for number in x]
    x2 = [a * b for a, b in zip(x, x1)]
    x3 = [obj._lamda * number for number in x2]
    return x3
