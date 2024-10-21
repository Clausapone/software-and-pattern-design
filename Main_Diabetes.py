from CreateEmbeddings import CreateEmbeddings
import pandas as pd
import owlready2 as owl

dataset_path = "diabetes.csv"
ontology_path = "diabetes_ontology.rdf"

embeddings_maker = CreateEmbeddings(dataset_path, ontology_path)
OWL_embeddings = embeddings_maker.fit()

dataset = pd.read_csv(dataset_path)

New_dataset = dataset.concatenate(OWL_embeddings)   # concatenazione orizzontale per ottenere il nuovo dataset arricchito
