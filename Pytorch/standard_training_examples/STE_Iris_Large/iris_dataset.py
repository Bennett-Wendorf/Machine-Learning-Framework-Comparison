import os
import pandas as pd
import torch
from torch.utils.data import Dataset

IRIS_LABEL_INDEX = 4
DEFAULT_TRAIN_PERCENT = 0.8

class IrisDataset(Dataset):
    def __init__(self, data_file, transform=None, target_transform=None):
        self.data = pd.read_csv(data_file)
        self.labels = self.data.iloc[:, IRIS_LABEL_INDEX].unique()
        self.labels = {label: idx for idx, label in enumerate(self.labels)}
        self.label_lookup = {idx: label for label, idx in self.labels.items()}
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datapoints = self.data.iloc[idx, 0:IRIS_LABEL_INDEX]
        label = self.labels[self.data.iloc[idx, IRIS_LABEL_INDEX]]

        if self.transform:
            datapoints = self.transform(datapoints, dtype=torch.float32)
        if self.target_transform:
            label = self.target_transform(label)
        
        return datapoints, label

    def get_from_train_test_splits(self, train_percent=DEFAULT_TRAIN_PERCENT):
        train_size = int(train_percent * len(self))
        test_size = len(self) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(self, [train_size, test_size])
        return train_dataset, test_dataset

    def get_label(self, idx):
        return self.label_lookup[idx]

    def get_index(self, label):
        return self.labels[label]

    def get_all_datapoints_and_labels(self):
        return self.data.iloc[:, 0:IRIS_LABEL_INDEX], self.data.iloc[:, IRIS_LABEL_INDEX]

    def get_all_datapoints_and_label_indexes(self):
        data, labels = self.get_all_datapoints_and_labels()
        return data, labels.apply(lambda label: self.get_index(label))