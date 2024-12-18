import numpy as np
import pandas as pd
import owlready2 as owl
from owlready2 import sync_reasoner_pellet
from rdflib import Graph
import networkx as nx
from node2vec import Node2Vec

# {SCRIPT TO SAVE NUMPY EMBEDDINGS COMPUTED FROM OUR INPUT ONTOLOGY OFFLINE (using Node2Vec)}

# UTILS FUNCTIONS ------------------------

# Function to populate input ontology using dataset information
def data_2_owl(dataset, features_list, onto):

    with (onto):
        individual_class = onto[features_list[0]]   # Main_individual class
        for data_sample in dataset:
            for i, feature in enumerate(features_list):
                if i == 0:
                    individual = individual_class(data_sample[i])   # Adding individuals to Main_individual class
                else:
                    hasFeature = 'has' + feature    # Adding Data properties to each individual
                    setattr(individual, hasFeature, [data_sample[i]])


# Function to deduct new axioms in our input ontology with Pellet reasoner
def onto_reasoning(onto):
    owl.reasoning.JAVA_MEMORY = 8000  # 8 GB of memory allocated for reasoning
    try:
        with onto:
            sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
            onto.save("reasoned_diabetes_ontology.rdf")
    except:
        print('inference Failed')


# Function to extract individual embeddings from our trained Node2Vec model given individual IRI
def owl_2_embedding(model, base_iri):
    individual_iri = base_iri + data_sample[0]  # computing individual IRI
    return np.array(model.wv[individual_iri])



# EMBEDDING CREATION PROCESS ------------------------

# Configuration
dataset_path = "diabetes2000.csv"
ontology_path = "diabetes_ontology.rdf"
embedding_size = 15

# Ontology population
onto = owl.get_ontology(ontology_path).load()
dataframe = pd.read_csv(dataset_path).iloc[:, :-1]      # we don't take into account last column (Outcomes)
features_list = np.array(dataframe.columns)
dataset = dataframe.to_numpy()

data_2_owl(dataset, features_list, onto)
populated_ontology_path = "populated_" + ontology_path
onto.save(populated_ontology_path)

# Pellet reasoning
onto_reasoning(onto)

# Graph creation
rdf_graph = Graph()     # creating a graph parsing a rdf file
rdf_graph.parse("reasoned_diabetes_ontology.rdf")

nx_graph = nx.Graph()   # transformation into networkx graph adding nodes (subjcects-s, objects-o) and edges (predicates-p)
for s, p, o in rdf_graph:
    nx_graph.add_edge(s, o, label=p)

# Node2Vec instantiation
node2vec = Node2Vec(nx_graph,
                    dimensions=embedding_size,
                    walk_length=20,
                    num_walks=1000,
                    workers=8,
                    p=1,
                    q=4)

model = node2vec.fit()

# Embedding extraction
OWL_dataset = np.empty((0, embedding_size))
base_iri = onto.base_iri
for data_sample in dataset:     # computing embeddings iterating over dataset rows (individuals)
    individual_embedding = owl_2_embedding(model, base_iri)
    OWL_dataset = np.vstack((OWL_dataset, individual_embedding))

# Saving embeddings offline in a numpy file
np.save("OWL_dataset2000_N2V.npy", OWL_dataset)
