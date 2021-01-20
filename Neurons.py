''' Using numpy library only for data pre-processing'''
'''TODO: Explain the purpose of this class
         fix RMSE function
         fix predict function'''
import numpy as np
import random
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
from MatUtils import transpose, dot, add, sigmoid, sigmoid_derivative


class Neurons:

    def __init__(self, config):
        super().__init__()
        '''Initializing the Neural Network with single hidden layer
        * 3 Neurons nodes in the hidden layer
        * 450 Epochs
        * Learning rate of o.2
        * Momentum of 0.7
        * Lambda 0.1: Setting higher value of Lambda leads to buffer overflow when passed to activation function
        '''
        self._hiddenNeurons = config["hiddenNeurons"]
        self._epochs = config["epochs"]
        self._learning_rate = config["learning_rate"]
        self._lamda = config["lamda"]
        self._momentum = config["momentum"]
        self.inputs = []
        self.output = []
        self.labelOutput = []
        self.predicted = []
        self.hidden = []
        self.output = []
        '''minMAx list stores the min and max value of each column used for normalization of the respective columns.
        These values are stored and printed, which is used in the NeuralNetHolder class of the pygameto normalize 
        and denormalize input and output respectively'''
        self.minMax = []
        random.seed(1)
        '''Initializing weights and biases of Input and )utput layer with a random values between 0 & 1.
           Shape of input and output parameters depends on the network configuration.
           Lander game has two inputs (X-distance, Y-distance) and 2 outputs(X-Velocity,Y-Velocity).
           Therefore the network is configured with 2 Neurons at the input layer and 2 Neurons at the output layer.
           Number of Hidden layer Neurons is a hyper parameter and it is needed to be tuned for optimal output.
           Shape of Neurons and weights at different layers:
           Input neurons: (1,2)
           Output Neurons: (1,2)
           Hidden Layer Neurons: (2, 'hiddenNeurons')
           Input Weights: (2, 'hiddenNeurons')
           Input Bias: (1, 'hiddenNeurons')
           Output Weights: ('hiddenNeurons', 2)
           Output Bias: (1,2)
        '''
        inputNeurons, outputNeurons = 2, 2
        self.input_weights = [[round(random.random(), 8) for e in range( self._hiddenNeurons)] for e in range(inputNeurons)]
        self.output_weights = [[round(random.random(), 8) for e in range(outputNeurons)] for e in range( self._hiddenNeurons)]
        self.input_bias = [round(random.random(), 8) for e in range( self._hiddenNeurons) for e in range(1)]
        self.output_bias = [round(random.random(), 8) for e in range(outputNeurons) for e in range(1)]
        '''Initializing all the old weights and delta weights with zero.
           Shape of these weights is same as their respective current weights'''
        self.oldIpWeights = [[0 for e in range( self._hiddenNeurons)] for e in range(inputNeurons)]
        self.oldOpWeights = [[0 for e in range(outputNeurons)] for e in range( self._hiddenNeurons)]
        self.oldIpDw = [[0 for e in range( self._hiddenNeurons)] for e in range(inputNeurons)]
        self.oldOpDw = [[0 for e in range(outputNeurons)] for e in range( self._hiddenNeurons)]
        self.oldIpBiasDw = [0 for e in range( self._hiddenNeurons) for e in range(1)]
        self.oldOpBiasDw = [0 for e in range(outputNeurons) for e in range(1)]

    '''This a loss function, used to calculate the root mean square error between the labelled output and the predicted output.
    Parameters:
    [y_labelled]:
    [y_predicted]
    Return Value:
    [root_mean_square_error]
    '''
    def root_mean_squared_error(self, y_labelled, y_predicted):
        square_error = [    (labelled - predicted) ** 2 for labelled, predicted in zip(y_labelled, y_predicted)]
        root_mean_square_error = sqrt(sum(square_error) / len(square_error))
        return root_mean_square_error

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
        layer_output = sigmoid(self, layer_activation)
        return layer_output

    '''This function does the forward propagation for the network.
        It calculates the value of neuron nodes at Hidden layer and this is then used to predict
        the output neuron nodes at the output layer.
        Parameters:
        [inputs]:     It can be input data from the training,validate or test data set.
        [outputs]:    It is the labelled output data.
        Return Value: This function does not return anything but updates the class member variable 
                      for hidden and output layer nodes
    '''
    def feedForwardPass(self, inputs, outputs):
        self.labelOutput = outputs
        self.inputs = inputs
        self.hidden = self.calculateOutput(self.input_weights, self.inputs, self.input_bias)
        self.output = self.calculateOutput(self.output_weights, self.hidden, self.output_bias)

    '''This function calculates the local gradient at both output layer and hidden layer.
        It is called during the back propagation process, so the local gradient at output layer is
        calculated first and then at the hidden layer. It calculated using the following formula:
                local Gradient = lambda * nodes_value * (1 - nodes_value) * error
        Optimizes the cost function, which calculates optimal values for the hyper parameters.
        Parameters:
        [nodes_value]: It is either the nodes value at the output layer or at the hidden layer, depending 
                       on the layer it is called.
        [err]:         It either the error value at the output layer or at the hidden layer, depending
                       on the 
                       layer it is called.
        Return Value:  Returns the local gradient
    '''
    def calculateLocalGradient(self, x, err):
        localGrad = [a * b for a, b in zip(sigmoid_derivative(self, x), err)]
        return localGrad

    '''This function calculates the delta weights and adds it to the respective weight at current iteration.
    Delta Weight at iteration/time 't' is calculated using the formula:
            delta_weight(t) = learning_rate * Local_gradient * nodes_value + momentum * delta_weights(t-1)
            Weights = weights + delta_weights(t)
    This function is called to update 1. Input layer weights and bias
                                      2. Output Layer weights and bias
    Parameters:
    [localGrad]      : Input or output local gradient, depending on  the layer it is called.
    [weights]        : It can be input and output: weights or bias weights.
    [oldDeltaWeights]: Delta weight at iteration/time t-1, where is the current iteration/time.
                       layer it is called.
    [nodes_value]    : It is either hidden layer nodes value or input layer nodes value,
                       depending on where it is called. It is passed 0 when the function is called to update the
                       bias weights.
    [bias]           : Flag to differentiate if the weight update is for input/output weights or bias weights.
                       The default value is set to False.
    Return Value     : Returns the updated weights and delta weights at current iteration.
    '''
    def weightUpdate(self, localGrad, weights, oldDeltaWeights, data=0, bias=0):
        deltaWeights = []
        if bias:  # CALCULATE BIAS WEIGHTS, FOR BIAS DATA(INPUT) IS 0
            lg_sum = sum(localGrad)
            m = [self._momentum * number for number in oldDeltaWeights]
            n = self._learning_rate * lg_sum
            deltaWeights = [n + y for y in m]
        else:
            m = []
            k = []
            for i in localGrad:
                for j in data:
                    m.append(i * j)
                k.append(m[:])
                m.clear()
            n = [[self._momentum * number for number in y] for y in oldDeltaWeights]
            deltaWeights = add(n, transpose(k))

        weights = add(weights, deltaWeights)
        return weights, deltaWeights

    '''This function performs the back propagation of the neural network.
        It does the following tasks:
        * Calculates Error between predicted output and Labelled output
        * Calculates Output local gradient.
        * Calculates the matrix product of output local gradient and out weights at time t-1
        * Calculates input local gradient
        * update input/output layer node and bias weights.
    There are not parameters passed to this function and it does not return any value.
    All the relevant weights are updated for the Neurons class weights member variables
    '''
    def backPropogation(self):
        """Error calculation at output layer"""
        err = [a - b for a, b in zip(self.labelOutput, self.output)]
        opLocalGrad = self.calculateLocalGradient(self.output, err)
        inErr = dot(transpose(opLocalGrad), self.oldOpWeights)
        self.oldOpWeights = self.output_weights
        ipLocalGrad = self.calculateLocalGradient(self.hidden, inErr)
        '''Weight and bias weight update at input and output layer, current iteration delta weight is stored as
        the old delta weight for the next iteration. 
        '''
        self.output_weights, self.oldOpDw = self.weightUpdate(opLocalGrad, self.output_weights, self.oldOpDw, self.hidden)
        self.output_bias, self.oldOpBiasDw = self.weightUpdate(opLocalGrad, self.output_bias, self.oldOpBiasDw, 0, 1)
        self.input_weights, self.oldIpDw = self.weightUpdate(ipLocalGrad, self.input_weights, self.oldIpDw,
                                                         self.inputs)
        self.input_bias, self.oldIpBiasDw = self.weightUpdate(ipLocalGrad, self.input_bias, self.oldIpBiasDw, 0, 1)
    
    '''This function does the validation on the validation set. It is called after every epoch to validate
       the performance of the network. It calls the feedforward function passing one row of input at a time.
       The predicted output is used to calculate the loss and RMSE is stored to plot it against epochs.
       This is used later to calculate the early stopping criteria.
       Parameters:
       [validate_input] :  validation input data set used for predictions.
       [validate_output] : validation labelled output data set used for calculating the loss.
       Return Value:
       [rmse_validate]: Returns RMSE for validation data set.
    '''
    def validate(self, inputsValidate, initialOpValidate):
        rmse_validate = []
        predicted_row = []
        for i in range(len(inputsValidate)):
            validIp = inputsValidate[i][:2]
            validIp = [float(i) for i in validIp]
            # print(trainIp)
            validOp = initialOpValidate[i][:2]
            validOp = [float(i) for i in validOp]
            # print(trainOp)
            self.feedForwardPass(validIp, validOp)
            predicted_row.append(self.output)
        rmse_validate.append(sqrt(mean_squared_error(initialOpValidate, predicted_row)))
        # print(self.root_mean_squared_error(initialOpValidate, predicted_row))
        return rmse_validate[-1]

    '''This function does the Predictions on the Test data set. It is called after training the network to test
       the performance of the network, it is called on the unseen data set. It calls the feedforward function passing 
       one row of input at a time.
       The predicted output is used to evaluate the performance of the network..
       Parameters:
       [xtest] :  Test input data set used for predictions.
       [ytest] :  Test labelled output data set used for calculating the loss.
       Return Value:
       [rmse_test]: Returns RMSE for Test data set.
    '''
    def predict(self, xtest, ytest):
        self.inputs = xtest
        rmse_test = 0
        predicted_row = []
        for row in range(len(xtest)):
            testIp = xtest[row][:2]
            testIp = [float(i) for i in testIp]
            # print(trainIp)
            testOp = ytest[row][:2]
            testOp = [float(i) for i in testOp]
            self.feedForwardPass(testIp, testOp)
            predicted_row.append(self.output)
        rmse_test = sqrt(mean_squared_error(ytest, predicted_row))
        print("RMSE Test", rmse_test)
        return rmse_test

    '''This function trains the network on the training data set. It calls the feedforward function passing 
       one row of input at a time.
       The back propagation is called after every 512 rows of input, the weights are not updated after every 
       forward propagation but they are updated after every 512 times the network is trained with input row
       in one epoch. Epochs are configured as a hyper parameter to tune the network.
       The predicted output is used to evaluate the performance of the network..
       Parameters:
       [inputsTotal] :     Train input data set used for training.
       [initialOpTotal] :  Train labelled output data set used for calculating the loss.
       [x_validate]     :  Validate input data set used for training.
       [y_validate]     :  Validate labelled output data set used for calculating the loss.
       Return Value     :  This function does not return anything.
    '''
    def train(self, inputsTotal, initialOpTotal, x_validate, y_validate):
        rmse_train = []
        rmse_validate = []
        predicted_row = []
        rmse_prev_10 = [1]  # Initialized this to Zero to handle 'divide by zero' exception
        for itr in range(self._epochs):
            # print("EPOCH: ", itr)
            for row in range(len(inputsTotal)):
                trainIp = inputsTotal[row][:2]
                trainIp = [float(i) for i in trainIp]
                # print(trainIp)
                trainOp = initialOpTotal[row][:2]
                trainOp = [float(i) for i in trainOp]
                # print(trainOp)
                self.feedForwardPass(trainIp, trainOp)
                predicted_row.append(self.output)
                ''' Back propagation is only done after every 256 rows of input data.
                This makes the procesing fast.'''
                if row % 256 == 0:
                    self.backPropogation()
                # print(self.root_mean_squared_error(initialOpTotal, predicted_row))
            rmse_train.append(sqrt(mean_squared_error(initialOpTotal, predicted_row)))


            predicted_row.clear()
            '''Calling the validate function to predict the network performance on the traine network after all the
            inputs rows are fed to forward propagation and weight is updated.'''
            rmse_validate.append(self.validate(x_validate, y_validate))
            print("Epoch:", itr, " ................... RMSE TRAIN:", rmse_train[-1], " ............... RMSE Validate:",
                  rmse_validate[-1])
            # Logic to predic Early Stopping criteria.
            # Early stopping criteria is only activated after 550 epochs.
            if itr > 550:
                ''' checking if current rmse is greater than average of last 10 rmse values.
                If it is greater than training of the network is stopped to avoid overfitting.'''
                if rmse_validate[-1] > (sum(rmse_prev_10) / len(rmse_prev_10)):
                    print("Average", sum(rmse_prev_10) / len(rmse_prev_10))
                    print("rmse", rmse_validate[-1])
                    print("Early stopping")
                    break
                if len(rmse_prev_10) > 10:
                    rmse_prev_10.pop(0)
                rmse_prev_10.append(rmse_validate[-1])
        plt.plot(range(self._epochs), rmse_train, label='Train')
        plt.plot(range(self._epochs), rmse_validate, label='Validate')
        plt.legend()
        plt.show()

        print("Final Input weights: ", end='')
        print(self.input_weights, sep=",")
        print("Final Input bias: ", end='')
        print(self.input_bias, sep=",")
        print("Final output weights: ", end='')
        print(self.output_weights, sep=",")
        print("Final output bias: ", end='')
        print(self.output_bias, sep=",")
        print("min max values", self.minMax)
