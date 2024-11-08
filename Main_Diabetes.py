import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.nn import BCELoss

import pandas as pd
import numpy as np

from Train import train
from Test import test
from Model import NeuralNetwork
from Metrics_plot import show_metrics, show_loss_history, show_confusion_matrix

torch.manual_seed(42)

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
#X = dataset

# preprocessig
scaler = StandardScaler()
X = scaler.fit_transform(X)

model = NeuralNetwork(input_dim=X.shape[1])
loss_criterion = BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

X = torch.tensor(X, dtype=torch.float).to(device)
Y = torch.tensor(Y, dtype=torch.float).to(device)

# training
train_mask, test_mask = train_test_split(np.arange(X.shape[0]), test_size=0.2, random_state=42, shuffle=True)

loss_history = train(model, X, Y, train_mask, optimizer, loss_criterion, 1500)

# test
test_loss, test_accuracy, test_precision, test_recall, test_f1_s, conf_mat = test(model, X, Y, test_mask, loss_criterion)

show_loss_history(loss_history)
show_confusion_matrix(conf_mat)
show_metrics(test_accuracy, test_precision, test_recall, test_f1_s)

