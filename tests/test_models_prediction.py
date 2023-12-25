import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

class TestRegressionLogistic(unittest.TestCase):
    def setUp(self):
        with open('../models/regression_logistic_model.pck', 'rb') as f:
            self.loaded_model, self.loaded_sc = pickle.load(f)

    def test_prediction(self):
        test_data = np.array([[1.4, 0.2],
                              [1.7, 0.2],
                              [5.3, 2.3],
                              [5.7, 2.1],
                              [1.2, 0.2],
                              [5.6, 2.4],
                              [6.6, 2.1],
                              [1.2, 0.2],
                              [5.8, 1.6],
                              [4.6, 1.3],
                              [3.3, 1. ],
                              [3.9 ,1.4]])

        test_data_std = self.loaded_sc.transform(test_data)

        predictions = self.loaded_model.predict(test_data_std)

        expected_values = np.array([0, 0, 2, 2, 0, 2, 2, 0, 2, 1, 1, 1])  

        np.testing.assert_array_equal(predictions, expected_values)

class TestSVM(unittest.TestCase):
    def setUp(self):
        with open('../models/svm_model.pck', 'rb') as f:
            self.loaded_model, self.loaded_sc = pickle.load(f)

    def test_prediction(self):
        test_data = np.array([[1.4, 0.2],
                              [1.7, 0.2],
                              [5.3, 2.3],
                              [5.7, 2.1],
                              [1.2, 0.2],
                              [5.6, 2.4],
                              [6.6, 2.1],
                              [1.2, 0.2],
                              [5.8, 1.6],
                              [4.6, 1.3],
                              [3.3, 1. ],
                              [3.9 ,1.4]])

        test_data_std = self.loaded_sc.transform(test_data)

        predictions = self.loaded_model.predict(test_data_std)

        expected_values = np.array([0, 0, 2, 2, 0, 2, 2, 0, 2, 1, 1, 1])  

        np.testing.assert_array_equal(predictions, expected_values)

class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        with open('../models/decision_tree_model.pck', 'rb') as f:
            self.loaded_model, self.loaded_sc = pickle.load(f)

    def test_prediction(self):
        test_data = np.array([[1.4, 0.2],
                              [1.7, 0.2],
                              [5.3, 2.3],
                              [5.7, 2.1],
                              [1.2, 0.2],
                              [5.6, 2.4],
                              [6.6, 2.1],
                              [1.2, 0.2],
                              [5.8, 1.6],
                              [4.6, 1.3],
                              [3.3, 1. ],
                              [3.9 ,1.4]])

        test_data_std = self.loaded_sc.transform(test_data)

        predictions = self.loaded_model.predict(test_data_std)

        expected_values = np.array([0, 0, 2, 2, 0, 2, 2, 0, 2, 1, 1, 1])  

        np.testing.assert_array_equal(predictions, expected_values)     

class TestKnn(unittest.TestCase):
    def setUp(self):
        with open('../models/knn_model.pck', 'rb') as f:
            self.loaded_model, self.loaded_sc = pickle.load(f)

    def test_prediction(self):
        test_data = np.array([[1.4, 0.2],
                              [1.7, 0.2],
                              [5.3, 2.3],
                              [5.7, 2.1],
                              [1.2, 0.2],
                              [5.6, 2.4],
                              [6.6, 2.1],
                              [1.2, 0.2],
                              [5.8, 1.6],
                              [4.6, 1.3],
                              [3.3, 1. ],
                              [3.9 ,1.4]])

        test_data_std = self.loaded_sc.transform(test_data)

        predictions = self.loaded_model.predict(test_data_std)

        expected_values = np.array([0, 0, 2, 2, 0, 2, 2, 0, 2, 1, 1, 1])  

        np.testing.assert_array_equal(predictions, expected_values)              

if __name__ == '__main__':
    unittest.main()