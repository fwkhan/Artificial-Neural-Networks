'''
This is the main file for the training and prediction of the Lander pygame.
'''

from Neurons import Neurons
import DataDigger

'''
This is the main loop and it performs followings tasks:
        * Reads the data file from disk.
        * Instantiates an object of Neurons Class and
          configures Network Parameters.
        * Instantiates an object of DataDigger, used for pre-processing of data.
        * Trains the Networks and performs Prediction on the test data.
'''
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    '''Assigning names to data columns to beread from the disk'''
    names = ["X distance", "Y distance", "Vel Y", "Vel X"]
    file_path = r"C:\Users\faiza\Desktop\ce889_dataCollection-old.csv"

    '''config dictionary object, initializes Neurons class with the following hyper parameters
        * Epochs: Number of cycles the network is trained with entire training set.
        * Learning Rate: Gives the Step size to update the weights
        * Momentum: Needed for convergence of the network.
        * HiddenNeurons: The number of neuron nodes in the hidden layer.
        Performance of the network depends on the proper tuning of these Hyper Parameters.'''
    config = {"epochs": 450, "learning_rate": 0.2, "lamda": 0.1, "momentum": 0.7, "hiddenNeurons": 3 }
    nn = Neurons(config)
    dd = DataDigger.DataDig()
    dd.readData(file_path, names)
    ''' Train-Validate split of the entire data with 80% Train data and 20% Test data'''
    validate_size = 0.2
    dd.splitDataToTrainAndStore(validate_size)
    ''' Test-Validate split of the Validate data with 50% Test data and 50% Validate data'''
    test_size = 0.5
    dd.splitStoreToValidateAndTest(test_size)
    '''Normalized data to train and validate'''
    x_train, y_train, x_validate, y_validate, x_test, y_test  = dd.getNormalizeData()
    ''' Calling the train method of NeuralNetwork class to train the network with train data set'''
    nn.train(x_train, y_train, x_validate, y_validate)
    '''Calling the predict method of the class NeuralNetwork to test the trained Network'''
    nn.predict(x_test, y_test)
