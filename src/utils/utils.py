"placeholder for various utils functions"

import json
import csv
import pandas as pd
from pathlib import Path

# example function to load data from a specified directory
def load_dataset(question_name):
    base_path = Path("../data") / question_name
    result = {}
 
    for file_path in base_path.glob("*"):
        stem = file_path.stem
        suffix = file_path.suffix.lower()
 
        try:
            if suffix == '.json':
                with open(file_path, 'r') as f:
                    result[stem] = json.load(f)
            elif suffix == '.csv':
                with open(file_path, 'r') as f:
                    result[stem] = list(csv.DictReader(f))
            else:
                with open(file_path, 'r') as f:
                    result[stem] = f.read()
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
 
    return result

# example function to save model results in a specified directory
def save_model_results():
    """Placeholder for save_model_results function."""
    pass

# example function to plot data from a specified directory
def plot_data():
    """Placeholder for plot_data function."""
    pass

def create_demand_profiles(type_of_demand):
    total_demand = 24 #KWh over the day
    demand_profiles = {}
    for type in type_of_demand:
        if type == 'industrial':
            demand = [1 for i in range(24)]
            demand_profiles[type]=demand
        if type == 'office':
            demand = [0,0,0,0,0,0,0,0.25,0.5,0.75,1,1,0.5,0.5,1,1,1,0.75,0.5,0.5,0,0,0,0]
            s = sum(demand)
            demand_profiles[type]=[d*total_demand/s for d in demand]
    return demand_profiles