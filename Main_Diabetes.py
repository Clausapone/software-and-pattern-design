from CreateEmbeddings import CreateEmbeddings
import pandas as pd
import numpy as np

dataset_path = "toy_diabetes.csv"     #SU: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
dataset = np.array(pd.read_csv(dataset_path).iloc[:, 1:])   # non considero la prima feature poichè rappresenta l'ID univoco del

ontology_path = "diabetes_ontology.rdf"   #SU: https://bioportal.bioontology.org/ontologies/CDMONTO/?p=classes&conceptid=http%3A%2F%2Fwww.semanticweb.org%2Fkhaled%2Fontologies%2F2024%2F7%2FCDMOnto%23Diabetes&lang=en
embeddings_maker = CreateEmbeddings(dataset_path, ontology_path)
OWL_dataset = embeddings_maker.fit()

# arricchimento del dataset iniziale
New_dataset = np.hstack((dataset, OWL_dataset))

# preprocessig (ricorda i dummies)

# cross validation

# training

# test
