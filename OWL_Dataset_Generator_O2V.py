import configparser

import numpy as np
import pandas as pd
import owlready2 as owl

from owl2vec_star import owl2vec_star

# {SCRIPT TO SAVE NUMPY EMBEDDINGS COMPUTED FROM OUR INPUT ONTOLOGY OFFLINE (using OWL2Vec*)}

# UTILS FUNCTIONS

# Function to populate input ontology using dataset information
def data_2_owl(dataset, features_list, onto):

    with (onto):
        main_individual_class = onto[features_list[0]]  # Main_individual class
        for data_sample in dataset:
            for i, feature in enumerate(features_list):
                if i == 0:
                    individual = main_individual_class(data_sample[i])      # Adding individuals to Main_individual class
                else:
                    hasFeature = 'has' + feature        # Adding Data properties to each individual
                    setattr(individual, hasFeature, [data_sample[i]])


# Function to extract individual embeddings from our trained Node2Vec model given individual IRI
def owl_2_embedding(model, base_iri):
    individual_iri = base_iri + data_sample[0]      # computing individual IRI
    return np.array(model.wv[individual_iri])


# EMBEDDING CREATION PROCESS

# Configuration
dataset_path = "diabetes2000.csv"
ontology_path = "diabetes_ontology.rdf"
config_file_path = "my_config.cfg"

# Ontology population
onto = owl.get_ontology(ontology_path).load()
dataframe = pd.read_csv(dataset_path).iloc[:, :-1]
features_list = np.array(dataframe.columns)
dataset = dataframe.to_numpy()

data_2_owl(dataset, features_list, onto)
onto.save("populated_" + ontology_path)

# Parsing config file to retrieve embedding size
config = configparser.ConfigParser()
config.read(config_file_path)
embedding_size = int(config['MODEL']['embed_size'])

# OWL2Vec* instantiation
model = owl2vec_star.extract_owl2vec_model(ontology_file=config['BASIC']['ontology_file'],
                                           config_file=config_file_path,
                                           uri_doc=True,
                                           lit_doc=True,
                                           mix_doc=True)

# Embedding extraction
OWL_dataset = np.empty((0, embedding_size))
base_iri = onto.base_iri
for data_sample in dataset:     # computing embeddings iterating over dataset rows (individuals)
    individual_embedding = owl_2_embedding(model, base_iri)
    OWL_dataset = np.vstack((OWL_dataset, individual_embedding))

# Saving embeddings offline in a numpy file
np.save("OWL_dataset2000_O2V.npy", OWL_dataset)
