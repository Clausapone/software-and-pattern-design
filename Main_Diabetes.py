from Data2OWL import Data2OWL
import pandas as pd
import owlready2 as owl

dataset_path = "diabetes.csv"
ontology_path = "diabetes_ontology.rdf"

embeddings_creator = Data2OWL(dataset_path, ontology_path)
Embeddings = embeddings_creator.fit()

dataset = pd.read_csv(dataset_path)
New_dataset = dataset.concatenate(Embeddings)
