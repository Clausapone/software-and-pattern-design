import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.nn import BCELoss

import pandas as pd
import numpy as np

from Train import train
from Test import test
from Model import Simple_NeuralNetwork
from Metrics_plot import show_metrics, show_loss_history, show_confusion_matrix

torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# LOAD DEL DATASET E PREPROCESSING DEI DATI
dataset_path = "diabetes2000.csv"
dataframe = pd.read_csv(dataset_path)
dataframe.drop('Patient', axis=1, inplace=True)     # non considero la prima colonna con l'ID univoco del paziente

# alcune modifiche manuali al dataset
dataframe["Gender"] = dataframe["Gender"].map({"Male": 1, "Female": 0})
dataframe["Smoking_history"] = dataframe["Smoking_history"].map({"never": 0, "former": 1, "current": 2, "ever": 3})

Y = np.array(dataframe['Diabetes'])       # variabile target

dataframe.drop('Diabetes', axis=1, inplace=True)
dataset = np.array(dataframe)

OWL_dataset = np.load("OWL_dataset2000.npy")      # dataset OWL

# nuovo dataset arricchito
X = np.hstack((dataset, OWL_dataset))

model = Simple_NeuralNetwork(X.shape[1])
loss_criterion = BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# splitting
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float).to(device)
X_test = torch.tensor(X_test, dtype=torch.float).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float).to(device)

loss_history = train(model, X_train, Y_train, optimizer, loss_criterion, 1100)

# test
test_loss, test_accuracy, test_precision, test_recall, test_f1_s, conf_mat = test(model, X_test, Y_test, loss_criterion)

show_loss_history(loss_history)
show_confusion_matrix(conf_mat)
show_metrics(test_accuracy, test_precision, test_recall, test_f1_s)

