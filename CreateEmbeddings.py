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

        dataset = pd.read_csv(self.dataset_path).iloc[:, :-1]  # ottengo tutto il dataset dal path tranne la colonna target
        features_list = np.array(dataset.columns)
        dataset = dataset.to_numpy()

        onto = owl.get_ontology(self.ontology_path).load()  # ottengo l'ontologia dal path

        #OWL_dataset = matrice vuota

        for data_sample in dataset:
            new_ontology = self.data_2_owl(features_list, data_sample, onto)  # popolo ontologia
            embedding = self.owl_2_embedding(new_ontology)  # creo l'embedding dell'ontologia popolata
            OWL_dataset = concatena embedding alla matrice

#       return OWL_dataset

    # avendo i dati e l'ontologia vuota, restituisce l'ontologia popolata (usando owlready2)
    def data_2_owl(self, features_list, data_sample, onto):

        with onto:

            for i, feature in enumerate(features_list):
                feature_instance = onto[feature]  # istanza con lo stesso nome della feature
                feature_value = data_sample[i]  # valore della feature per il data sample passato a data_2_owl

                feature_name = feature_instance.name  # nome della feature
                feature_name_lower = feature_name.lower()  # nome della data property della feature (la poniamo in maniera standad come nome della feature in lower case per evitare la ridondanza)

                # se l'istanza non ha già una data property chiamata con la feature_name in lower case la creo e assegno il valore
                if not hasattr(feature_instance, feature_name_lower):
                    new_data_property = owl.types.new_class(feature_name_lower, (owl.DataProperty,))
                    new_data_property.range = [type(feature_value)]

                setattr(feature_instance, feature_name_lower, [feature_value])

        return onto



    # avendo l'ontologia popolata, restituisce un embedding (usando OWL2VEC)
    def owl_2_embedding(self, new_onto):

        return embedding 1x30


