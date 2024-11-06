import configparser
import numpy as np
import pandas as pd
import owlready2 as owl
from owl2vec_star import owl2vec_star


# SCRIPT CHE CONSENTE DI SALVARE (OFFLINE) GLI EMBEDDINGS OTTENUTI DALL'ONTOLOGIA OWL

#   {UTILS FUNCTIONS}
def data_2_owl(features_list, data_sample, onto, ontology_path):
    with onto:

        for i, feature in enumerate(features_list):
            feature_instance = onto[feature]  # istanza con lo stesso nome della feature nell'ontologia
            feature_value = data_sample[i]  # valore della feature per il data sample mappato ad una data property nell'ontologia

            feature_name = feature_instance.name  # nome della feature
            feature_name_lower = feature_name.lower()  # nome della data property della feature (la poniamo in maniera standad come nome della feature in lower case per evitare la ridondanza)

            # se l'istanza non ha già una data property chiamata con la feature_name in lower case la creo e assegno il valore
            if not hasattr(feature_instance, feature_name_lower):
                new_data_property = owl.types.new_class(feature_name_lower, (owl.DataProperty,))
                new_data_property.range = [type(feature_value)]

            setattr(feature_instance, feature_name_lower, [feature_value])

            onto.save("new_"+ontology_path)


# avendo l'ontologia popolata, restituisce un embedding (usando OWL2VEC)
def owl_2_embedding(ontology_path, config_file_path):
    model = owl2vec_star.extract_owl2vec_model(ontology_file="new_"+ontology_path, config_file=config_file_path, uri_doc=True, lit_doc=True, mix_doc=True)

    return np.array(model.wv['person']) # ?????????????????????????????????


#   {EMBEDDING CREATION PROCESS}
dataset_path = "toy_diabetes.csv"  #SU: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
ontology_path = "diabetes_ontology.rdf"  #SU: https://bioportal.bioontology.org/ontologies/CDMONTO/?p=classes&conceptid=http%3A%2F%2Fwww.semanticweb.org%2Fkhaled%2Fontologies%2F2024%2F7%2FCDMOnto%23Diabetes&lang=en
config_file_path = "my_config.cfg"

# carico l'ontologia
onto = owl.get_ontology(ontology_path).load()

# carico i nomi delle features e l'intero dataset senza tener conto della colonna target
dataset = pd.read_csv(dataset_path).iloc[:, :-1]
features_list = np.array(dataset.columns)
dataset = dataset.to_numpy()

# effettuo il parsing del file di configurazione per avere l'embedding size
config = configparser.ConfigParser()
config.read(config_file_path)
embedding_size = int(config['MODEL']['embed_size'])

# riempio il dataset OWL
OWL_dataset = np.empty((0, embedding_size))
for data_sample in dataset:
    data_2_owl(features_list, data_sample, onto, ontology_path)  # popolo ontologia
    embedding = owl_2_embedding(ontology_path, config_file_path)  # creo l'embedding dell'ontologia popolata
    OWL_dataset = np.vstack((OWL_dataset, embedding))

np.save("OWL_dataset.npy", OWL_dataset)
