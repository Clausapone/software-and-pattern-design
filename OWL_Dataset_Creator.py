import logging

import numpy as np
import pandas as pd
import owlready2 as owl
from owlready2 import sync_reasoner_pellet
from rdflib import Graph
import networkx as nx
from node2vec import Node2Vec
import owlready2

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
def owl_2_embedding(model, individual_iri):
    return np.array(model.wv[individual_iri])


# {EMBEDDING CREATION PROCESS} ----------------------------------------

# -----[configurazione]
dataset_path = "toy_diabetes.csv"  # SU: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
ontology_path = "diabetes_ontology-2.rdf"  # SU: https://bioportal.bioontology.org/ontologies/CDMONTO/?p=classes&conceptid=http%3A%2F%2Fwww.semanticweb.org%2Fkhaled%2Fontologies%2F2024%2F7%2FCDMOnto%23Diabetes&lang=en
embedding_size = 15

# -----[Popolo l'ontologia]

onto = owl.get_ontology(ontology_path).load()

# carico i nomi delle features e l'intero dataset senza tener conto della colonna target
dataframe = pd.read_csv(dataset_path).iloc[:, :-1]
features_list = np.array(dataframe.columns)
dataset = dataframe.to_numpy()

data_2_owl(dataset, features_list, onto)  # popolo ontologia
populated_ontology_path = "populated_" + ontology_path
onto.save(populated_ontology_path)

# -----[eseguo il reasoning]

owlready2.reasoning.JAVA_MEMORY = 8000    # attribuzione della memoria al reasoner, 8 GB
owlready2.set_log_level(9)

try:
    with onto:

        sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)  # aggiunta delle inferenze all'ontologia
        unsat = len(list(onto.inconsistent_classes()))     # controllo delle classi inconsistenti
        if unsat > 0:
            logging.warning("There are " + str(unsat) + " unsatisfiabiable classes.")
        print("Ontology successfully inferred using Pellet.")

except:
    print('Failet to use pellet for inference')



# -----[ottengo gli embeddings]


# grafo rdf dell'ontologia
rdf_graph = Graph()
rdf_graph.parse("prova_onto.rdf")

# trasformazione del grafo rdf in grafo networkx per node2vec
nx_graph = nx.Graph()

# popolamento del grafo
for s, p, o in rdf_graph:
    nx_graph.add_edge(s, o, label=p)

node2vec = Node2Vec(nx_graph, dimensions=embedding_size, walk_length=20, num_walks=200, workers=4)
model = node2vec.fit()

OWL_dataset = np.empty((0, embedding_size))
for data_sample in dataset:

    # iri dell'individuo di cui fare l'embedding
    individual_iri = onto.base_iri + data_sample[0]  # (ES. individual_iri=http://www.semanticweb.org/alessiomattiace/ontologies/2024/9/untitled-ontology-8#Patient_1)

    # calcolo l'embedding di ogni individuo dall'ontologia usando il suo iri
    individual_embedding = owl_2_embedding(model, individual_iri)
    OWL_dataset = np.vstack((OWL_dataset, individual_embedding))

np.save("OWL_dataset2000.npy", OWL_dataset)