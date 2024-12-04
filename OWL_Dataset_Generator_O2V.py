import configparser
import numpy as np
import pandas as pd
import owlready2 as owl

from owl2vec_star import owl2vec_star


# SCRIPT CHE CONSENTE DI SALVARE (OFFLINE) GLI EMBEDDINGS OTTENUTI DALL'ONTOLOGIA OWL


# {UTILS FUNCTIONS} ----------------------------------------
# funzione che popola l'ontologia avendo ontologia e dataset
def data_2_owl(dataset, features_list, onto):

    with (onto):
        # classe degli individui (ES. individual_class = classe Patient)
        individual_class = onto[features_list[0]]
        for data_sample in dataset:
            for i, feature in enumerate(features_list):
                if i == 0:
                    # creo un nuovo individuo di classe Paziente (ES. data_sample[0] = Patient_1)
                    individual = individual_class(data_sample[i])
                else:
                    hasFeature = 'has' + feature    # attributo per l'individuo (ES. hasFeature = hasAge)
                    setattr(individual, hasFeature, [data_sample[i]])


# avendo l'ontologia popolata, restituisce un embedding di un individuo usando OWL2VEC,
# l'individuo avrà il nome della prima feature
def owl_2_embedding(model, base_iri):
    # iri dell'individuo di cui fare l'embedding (ES. individual_iri=http://www.semanticweb.org/alessiomattiace/ontologies/2024/9/untitled-ontology-8#Patient_1)
    individual_iri = base_iri + data_sample[0]
    return np.array(model.wv[individual_iri])


# {EMBEDDING CREATION PROCESS} ----------------------------------------
# [Popolo l'ontologia]
dataset_path = "diabetes2000.csv"  #SU: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
ontology_path = "diabetes_ontology.rdf"  #SU: https://bioportal.bioontology.org/ontologies/CDMONTO/?p=classes&conceptid=http%3A%2F%2Fwww.semanticweb.org%2Fkhaled%2Fontologies%2F2024%2F7%2FCDMOnto%23Diabetes&lang=en
config_file_path = "my_config.cfg"

onto = owl.get_ontology(ontology_path).load()

# carico i nomi delle features e l'intero dataset senza tener conto della colonna target
dataframe = pd.read_csv(dataset_path).iloc[:, :-1]
features_list = np.array(dataframe.columns)
dataset = dataframe.to_numpy()

data_2_owl(dataset, features_list, onto)  # popolo ontologia
onto.save("populated_" + ontology_path)

# [ottengo gli embeddings]
# effettuo il parsing del file di configurazione per avere l'embedding size
config = configparser.ConfigParser()
config.read(config_file_path)
embedding_size = int(config['MODEL']['embed_size'])

# creazione del modello owl2vec
model = owl2vec_star.extract_owl2vec_model(ontology_file=config['BASIC']['ontology_file'],
                                           config_file=config_file_path,
                                           uri_doc=True,
                                           lit_doc=True,
                                           mix_doc=True)

# Creazione OWL_dataset
OWL_dataset = np.empty((0, embedding_size))
for data_sample in dataset:
    base_iri = onto.base_iri

    # calcolo l'embedding di ogni individuo dall'ontologia
    individual_embedding = owl_2_embedding(model, base_iri)
    OWL_dataset = np.vstack((OWL_dataset, individual_embedding))

np.save("OWL_dataset2000_O2V.npy", OWL_dataset)
