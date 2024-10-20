import numpy as np
import pandas as pd
import owlready2 as owl

"""
Funzione che prende in input un Dataset e un ontologia OWL e restituisce in output gli embeddings
dell'ontologia da concatenare al dataset in Main.py
"""

class Data2OWL:
    def __init__(self, dataset_path, ontology_path):
        self.dataset_path = dataset_path
        self.ontology_path = ontology_path

    def create_embedding(self, features_list, data_sample, ontology):

        for i, feature in enumerate(features_list):
            entity = ontology[feature]
            data_sample[i]

        return embedding 1x30


    def fit(self):

        dataset = pd.read_csv(self.dataset_path).iloc[:, :-1]       # per prendere tutto il dataset tranne l'outcome
        features_list = np.array(dataset.columns)
        ontology = owl.get_ontology(self.ontology_path).load()
        Matrice

        for data_sample in dataset:
            vettore = create_embedding(features_list, data_sample, ontology)
            Matrice.concatenate(vettore)

        return Matrice 4000x30