import pandas as pd
import numpy as np
import owlready2 as owl

from owlready2 import *
import random

# Load your ontology
onto = get_ontology("diabetes_ontology.rdf").load()

with onto:
    # Iterate over all instances in the ontology
    for instance in onto.individuals():
        # Get the instance name in lowercase
        instance_name_lower = instance.name.lower()

        # Create a new data property with the lowercase name
        # The property is added to the ontology's namespace
        new_data_property = types.new_class(instance_name_lower, (DataProperty,))

        # Optionally, set the domain and range of the property
        new_data_property.domain = [instance.is_a[0]]  # Assuming the first class in is_a
        new_data_property.range = [float]  # Assuming the random value is a float

        # Assign a random value to the data property for this instance
        random_value = random.random()  # Generates a random float between 0.0 and 1.0

        # Use setattr to set the property value for the instance
        setattr(instance, instance_name_lower, [random_value])

# Save the modified ontology
onto.save(file="modified_ontology.owl")
