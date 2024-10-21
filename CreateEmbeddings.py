import numpy as np
import pandas as pd
import owlready2 as owl

"""
Funzione che prende in input un Dataset e un ontologia OWL e restituisce in output gli embeddings
dell'ontologia da concatenare al dataset (in Main.py)
"""

class CreateEmbeddings:
    def __init__(self, dataset_path, ontology_path):
        self.dataset_path = dataset_path
        self.ontology_path = ontology_path


    # avendo dati e ontologia, restituisce la matrice di embeddings
    def fit(self):

        dataset = pd.read_csv(self.dataset_path).iloc[:, :-1]       # ottengo tutto il dataset dal path tranne la colonna target
        features_list = np.array(dataset.columns)

        onto = owl.get_ontology(self.ontology_path).load()      # ottengo l'ontologia dal path

        OWL_embeddings = matrice vuota

        for data_sample in dataset:
            new_ontology = self.data_2_owl(features_list, data_sample, onto)    # popolo ontologia
            embedding = self.owl_2_embedding(new_ontology)  # creo l'embedding dell'ontologia popolata
            OWL_embeddings = concatena embedding alla matrice

        return OWL_embeddings


    # avendo i dati e l'ontologia vuota, restituisce l'ontologia popolata
    def data_2_owl(self, features_list, data_sample, onto):

        for i, feature in enumerate(features_list):
            instance = onto[feature]
            value = data_sample[i]

            if instance

        return new_onto


    # avendo l'ontologia popolata, restituisce un embedding
    def owl_2_embedding(self, new_onto):

        return embedding 1x30

