import hashlib
import string
import itertools
import pickle
import csv
from sklearn.metrics import confusion_matrix

def write_dict_to_csv(data_dict, output_file):
    """
    Write rows from a dictionary to a single CSV file.

    Parameters:
    - data_dict: dict, where keys are arbitrary and values are lists or dictionaries representing rows.
    - output_file: str, the path of the output CSV file.

    Example:
    data_dict = {
        "row1": ["Name", "Age", "City"],
        "row2": ["Alice", 30, "New York"],
        "row3": ["Bob", 25, "Los Angeles"],
    }
    """
    # Check if the dictionary is empty
    if not data_dict:
        print("The dictionary is empty. Nothing to write.")
        return

    # Extract rows from the dictionary
    rows = list(data_dict.values())

    # Open the CSV file for writing
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write each row to the CSV file
        writer.writerows(rows)
    
    print(f"CSV file has been written to {output_file}.")

with open('rotom_testing/predicted_all.pickle', 'rb') as f:
    data_hash = pickle.load(f)

with open('rotom_testing/scores_all copy 7.pickle', 'rb') as f:
    data = pickle.load(f)
with open('rotom_testing/y_test_all.pickle', 'rb') as f:
    true_data = pickle.load(f)

print(data)


