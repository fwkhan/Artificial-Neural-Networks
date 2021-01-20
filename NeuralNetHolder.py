import pandas as pd
import numpy as np

lamda = 0.1

import math

"""This function does the transpose of a Matrix
[Parameters] : Input list which has to be transposed
Return value:
[result]     : Returns transposed list"""
def transpose(lst):
    # Logic for 2-D list
    if isinstance(lst[0], list):
        result = [list(x) for x in zip(*lst)]
    # Logic for 1-D list
    else:
        result = [list(x) for x in zip(lst)]
    return result


"""This function does the matrix multiplication of two Matrix
[list_1] : Input list 1
[list_2] : Input list 2
Return value:
[result]     : Returns matrix dot product"""
def dot(list_1, list_2):
    result = []  # final result
    col = 0
    for row in list_2:
        for a, b in zip(list_1, row):
            # if it is a 2-D list the converting list element to float
            if isinstance(a, list):
                a = a[0]
            col = col + a * b
        result.append(col)
        col = 0
    # result is copied as value to another list to avoid refrencing problem ,as result list is required be clear after
    # every loop.
    ret = result[:]
    result.clear()
    return ret


"""This function does the addition of two Matrix
[X] : Input list 1
[Y] : Input list 2
Return value:
[result]     : Returns matrix after adding each element"""
def add(X, Y):
    # 2-D list check
    if isinstance(X[0], list):
        result = [[c + d for c, d in zip(a, b)] for a, b in zip(X, Y)]
    else:
        result = [a + b for a, b in zip(X, Y)]
    return result


'''This class is used by pygame to perform the predictions on the input X-distance and Y-distance to
    predict X-velocity and Y-velocity.
    It uses weights obtained after training the network with the labelled output and input data.
    '''
class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        self.inputs = 0
        self.hidden = []
        self.output = []
        # Initializing input, output weights and biases with the weights and bias obtained after training the neural network.
        self.ipWeights =[[-4.195618143056083, -1.411419931029932, 8.692234783160464], [-0.865117890185862, 2.995644330969199, -0.08145339284729092]]
        self.opWeights =   [[1.7233464154179257, -5.466795613307096], [-2.208365133505515, -2.577302621011174], [0.9083546000844199, 7.6868053910382095]]

        self.ipBias =   [0.44893854617560147, -0.3112354838244117, 0.13204565617559097]
        self.opBias = [-0.5638511233525264, -1.0566289333525276]


    '''This is an activation function used for scaling the output between 0 & 1
    Mathematical formula used: 1 / 1+ exp**(-lamda*node_value)
    Parameters:
    [node_value]   : This can be either input neuron or hidden neuron value.
    Return Value:  : Normalized value of node_value'''

    def sigmoid(self, node_value):
        global lamda
        ret = [math.exp(-number * lamda) for number in node_value]
        ret = [1 + number for number in ret]
        ret = [1 / number for number in ret]
        return ret

    '''This function calculates the value of all the neurons in the hidden layer and the predicts the output
        layer's output. It gets the result of the Matrix dot product of weights and Neuron nodes at each layer, adds 
        the bias to it and passes it to an activation function, where it gets normalized to a value between 0 & 1.
        The activation function used here is a sigmoid.
        Parameters:
        [weights]      : It can be input data from the training,validate or test data set.
        [nodes_value]  : It is either the value of input nodes at input layer or the value of hidden nodes at hidden layer. 
        [biases]       : It is either the value of bias at input layer or the value of bias at hidden layer.  
        Return Value:
        [layer_output] : Returns values of hidden layer or output layer depending on the layer it is called.   
    '''
    def calculateOutput(self, weights, ip, biases):
        layer_activation = dot(ip, transpose(weights))
        layer_activation = add(layer_activation, biases)
        layer_output = self.sigmoid(layer_activation)
        return layer_output

    '''This function does the forward propagation for the network.
        It calculates the value of neuron nodes at Hidden layer and this is then used to predict
        the output neuron nodes at the output layer.
        Parameters:
        [inputs]:     It can be input data from the training,validate or test data set.
        Return Value: This function does not return anything but updates the class member variable 
                      for hidden and output layer nodes
    '''
    def feedForwardPass(self, inputs):
        self.inputs = inputs
        self.hidden = self.calculateOutput(self.ipWeights, self.inputs, self.ipBias)
        self.output = self.calculateOutput(self.opWeights, self.hidden, self.opBias)

    '''This function is called from gameloop to perform the predictions, it performs the following tasks:
        * Gets the input as float from the passed input_row string.
        * Normalizes the input data
        * Passes to feedforward for predicting X and Y velocity of the lander
        * Denormalizes the predicted output and pass it back as X and Y Velocity.
        ParametersL
        [input_rows]: X and Y distance passed as string
        [X-Velocity]: Predicted X-Velocity
        [Y-Velocity]: Predicted Y-Velocity'''
    def predict(self, input_row):
        '''Min and max values for all the 4 columns, which was used for normalize the data for training per column-wise
        using the same data to normalize the X-distance and Y-distance from the game.
        It is also used to denormalize the output X-Velocity and Y-velocity'''
        x1min, x1max, x2min, x2max, y1min, y1max, y2min, y2max = -811.2994749, 796.991646, 65.02206251, 805.5931615000001, -4.973630473, 8.0, -7.952574577999999, 7.979096635

        # Converting in the string input to float and stores it in a list.
        inputs = [float(el) for el in input_row .split(',')]
        # Normalizing both the inputs with respect to min and max values obtained from the training data.
        x1_Xdistance = (inputs[0] - x1min) / (x1max - x1min)
        x2_Ydistance = (inputs[1] - x2min) / (x2max - x2min)

        normalized_inputs = [x1_Xdistance, x2_Ydistance]
        denormalized_output = [0, 0]
        # Performing prediction on the normalized input data.
        self.feedForwardPass(inputs)
        # Denormalizing the predicted output
        denormalized_output[0] = self.output[0] * (y1max - y1min) + y1min
        denormalized_output[1] = self.output[1] * (y2max - y2min) + y2min

        self.output.clear()
        # Assigning X-Velocity and Y-Velocity with denormalized output
        VEL_Y = denormalized_output[0]
        VEL_X = denormalized_output[1]
        denormalized_output.clear()
        return VEL_X, VEL_Y
