import os
import pandas as pd
import numpy as np
import sklearn

IRIS_LABEL_INDEX = 4
DEFAULT_TRAIN_PERCENT = 0.8
LABEL_MAPPING = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

class IrisDataset():
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.labels = self.data.iloc[:, IRIS_LABEL_INDEX].unique()
        self.labels = {label: idx for idx, label in enumerate(self.labels)}
        self.label_lookup = {idx: label for label, idx in self.labels.items()}

    def __len__(self):
        return len(self.data)

    # TODO: Try to swap this out with sklearn's train_test_split
    def get_from_train_test_splits(self, train_percent=DEFAULT_TRAIN_PERCENT):
        self.data['Name'].replace(LABEL_MAPPING, inplace=True)
        train_dataset = self.data.sample(frac=train_percent)
        test_dataset = self.data.drop(train_dataset.index)
        return (
                train_dataset[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]].to_numpy(), 
                train_dataset["Name"].to_numpy()
            ), (
                test_dataset[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]].to_numpy(), 
                test_dataset["Name"].to_numpy()
            )

    def get_label(self, idx):
        return self.label_lookup[idx]

    def get_index(self, label):
        return self.labels[label]
    
    def get_all_datapoints_and_labels(self):
        return self.data.iloc[:, 0:IRIS_LABEL_INDEX], self.data.iloc[:, IRIS_LABEL_INDEX]

    def get_all_datapoints_and_label_indexes(self):
        data, labels = self.get_all_datapoints_and_labels()
        return data, labels.apply(lambda label: self.get_index(label))