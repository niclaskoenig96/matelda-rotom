import pickle
import pandas as pd

# Paths to the pickle and output CSV files
pickle_file_path = 'rotom_testing/tables_tuples.pickle'  # Update with the path to your pickle file
output_csv_path = 'rotom_testing/output.csv'  # Update with the desired output CSV file path

# Read the pickle file
with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

# Check the structure of the data and handle it appropriately
if isinstance(data, list):
    # If data is a list of dictionaries or tuples
    if all(isinstance(item, dict) for item in data):
        # Convert list of dictionaries directly to a DataFrame
        df = pd.DataFrame(data)
    elif all(isinstance(item, tuple) for item in data):
        # Reconstruct headers and rows for tuple-based data
        # Assuming the first tuple contains the header
        headers = data[0]
        rows = data[1:]
        df = pd.DataFrame(rows, columns=headers)
    else:
        raise ValueError("Unsupported data structure in the pickle file.")
elif isinstance(data, dict):
    # If data is a dictionary, treat keys as column names
    df = pd.DataFrame.from_dict(data, orient='index')
else:
    raise ValueError("Data type in pickle file is not suitable for reconstruction.")

# Save the reconstructed DataFrame to a CSV file
df.to_csv(output_csv_path, index=False)

print(f"The reconstructed CSV file has been saved to: {output_csv_path}")