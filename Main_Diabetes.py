import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch import optim
from torch.nn import BCELoss

import pandas as pd
import numpy as np

from Train import train
from Test import test
from Model import NeuralNetwork

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load del dataset standard e del datase ottenuto dall'ontologia OWL
dataset_path = "toy_diabetes.csv"
dataframe = pd.read_csv(dataset_path).iloc[:, 1:]   # non considero la prima feature poichè rappresenterà sempre l'ID univoco del paziente

Y = np.array(dataframe['Diabetes'])       # variabile target

dataframe.drop('Diabetes', axis=1, inplace=True)
dataframe = pd.get_dummies(data=dataframe, columns=['Gender', 'Smoking_history'], drop_first=True, dtype=int)
dataset = np.array(dataframe)

OWL_dataset = np.load("OWL_dataset.npy")      # dataset OWL

# nuovo dataset arricchito
X = np.hstack((dataset, OWL_dataset))

# preprocessig
scaler = RobustScaler()
X = scaler.fit_transform(X)

model = NeuralNetwork(input_dim=X.shape[1])
loss_criterion = BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

X = torch.tensor(X, dtype=torch.float).to(device)
Y = torch.tensor(Y, dtype=torch.float).to(device)

# training
train_mask, test_mask = train_test_split(np.arange(X.shape[0]), test_size=0.2, random_state=42, shuffle=True)
X_train, X_test = X[train_mask], X[test_mask]
Y_train, Y_test = Y[train_mask], Y[test_mask]

loss_history = train(model, X_train, Y_train, optimizer, loss_criterion, 1000)

# test
test_loss, test_accuracy, test_precision, test_recall, test_f1_s, conf_mat = test(model, X_test, Y_test, loss_criterion)
