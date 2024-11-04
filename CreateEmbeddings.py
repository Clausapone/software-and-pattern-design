import numpy as np
import pandas as pd
import owlready2 as owl
from owl2vec_star import owl2vec_star


"""
Funzione che prende in input un Dataset e un ontologia OWL e restituisce in output gli embeddings
dell'ontologia da concatenare al dataset (nel Main)
"""


class CreateEmbeddings:
    def __init__(self, dataset_path, ontology_path, embedding_size=30):
        self.dataset_path = dataset_path
        self.ontology_path = ontology_path
        self.embedding_size = embedding_size

    # avendo dati e ontologia, restituisce la matrice di embeddings
    def fit(self):

        dataset = pd.read_csv(self.dataset_path).iloc[:, :-1]  # ottengo tutto il dataset dal path tranne la colonna target
        features_list = np.array(dataset.columns)
        dataset = dataset.to_numpy()

        onto = owl.get_ontology(self.ontology_path).load()  # ottengo l'ontologia dal path

        OWL_dataset = np.empty((0, self.embedding_size))
        for data_sample in dataset:
            new_ontology = self.data_2_owl(features_list, data_sample, onto)  # popolo ontologia
            embedding = self.owl_2_embedding()  # creo l'embedding dell'ontologia popolata
            OWL_dataset = np.vstack((OWL_dataset, embedding))

        return OWL_dataset

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
                onto.save("new_diabetes_ontology.rdf")
        return onto

    # avendo l'ontologia popolata, restituisce un embedding (usando OWL2VEC)
    def owl_2_embedding(self):

        model = owl2vec_star.extract_owl2vec_model("./new_diabetes_ontology.rdf", "./my_config.cfg", True, True, True)

        return np.array(model.wv['person'])

