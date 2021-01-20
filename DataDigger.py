'''This file deals with the pre-processing of the input data.

Using Pandas, numpy and sklearn library for only pre-processing
of the data.'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

'''This class performs  the pre-processing of the data.
   Tasks performed:
   * Reading data stored on the disk in .csv format
   * Splits the read data into train data set and stores the remaining data
   * 70% data is the traininig data and 30% is stored.
   * Remaining 30% stored data is further split in 15% test and 15% validation data set'''


class DataDig:
    def __init__(self):
        self.X = None
        self.Y = None
        self.X_train = None
        self.Y_train = None
        self.X_store = None
        self.Y_store = None
        self.X_validate = None
        self.Y_validate = None
        self.X_test = None
        self.Y_test = None
        self.minMax = None

    def readData(self, file_path, names):
        df = pd.read_csv(file_path, names=names)
        df = df.dropna()
        self.X = df.iloc[:, 0:2].values
        self.Y = df.iloc[:, 2:4].values

    def splitDataToTrainAndStore(self, t_size):
        self.X_train, self.X_store, self.Y_train, self.Y_store = train_test_split(self.X, self.Y, test_size=t_size)
        print(self.X_train.shape, self.Y_train.shape)

    def splitStoreToValidateAndTest(self, t_size):
        self.X_validate, self.X_test, self.Y_validate, self.Y_test = train_test_split(self.X_store, self.Y_store,
                                                                                      test_size=t_size)

    def _normalize_data(self, data, minMax):
        x1_norm = ((data[0] - minMax[0]) / (minMax[1] - minMax[0]))
        x2_norm = ((data[1] - minMax[2]) / (minMax[3] - minMax[2]))
        y1_norm = ((data[2] - minMax[4]) / (minMax[5] - minMax[4]))
        y2_norm = ((data[3] - minMax[6]) / (minMax[7] - minMax[6]))

        x1_norm = x1_norm.reshape(len(data[0]), )
        x2_norm = x2_norm.reshape(len(data[0]), )
        in_data = np.vstack((x1_norm, x2_norm)).T

        y1_norm = y1_norm.reshape(len(data[0]), )
        y2_norm = y2_norm.reshape(len(data[0]), )
        out_data = np.vstack((y1_norm, y2_norm)).T

        in_data = in_data.tolist()
        out_data = out_data.tolist()
        return in_data, out_data

    def getNormalizeData(self):
        #              x1=[0]           x2=[1]           y1=[2]            y2=[3]
        train_data = [self.X_train[:, 0:1], self.X_train[:, 1:2], self.Y_train[:, 0:1], self.Y_train[:, 1:2]]
        # x1min,x1max,x2min,x2max,y1min,y1max,y2min,y2max
        self.minMax = [train_data[0].min(), train_data[0].max(), train_data[1].min(), train_data[1].max(),
                       train_data[2].min(), train_data[2].max(), train_data[3].min(), train_data[3].max()]
        x_norm_train, y_norm_train = self._normalize_data(train_data, self.minMax)

        validate_data = [self.X_validate[:, 0:1], self.X_validate[:, 1:2], self.Y_validate[:, 0:1],
                         self.Y_validate[:, 1:2]]
        x_norm_validate, y_norm_validate = self._normalize_data(validate_data, self.minMax)

        test_data = [self.X_test[:, 0:1], self.X_test[:, 1:2], self.Y_test[:, 0:1], self.Y_test[:, 1:2]]
        x_norm_test, y_norm_test = self._normalize_data(test_data, self.minMax)
        print("min max values", self.minMax)
        return x_norm_train, y_norm_train, x_norm_validate, y_norm_validate, x_norm_test, y_norm_test
