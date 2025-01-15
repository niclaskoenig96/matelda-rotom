import os
import pickle

def combine_pickles_in_directory(input_dir, output_filename):
    """
    Combines all pickle files in a directory into one list and saves it as a new pickle file.

    Args:
        input_dir (str): Path to the directory containing pickle files.
        output_filename (str): Path to the output pickle file.
    """
    combined_data = []

    # Iterate through all files in the directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.pickle'):
            file_path = os.path.join(input_dir, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                combined_data.append(data)

    # Save the combined data to a new pickle file
    with open(output_filename, 'wb') as f:
        pickle.dump(combined_data, f)

# Example usage
if __name__ == "__main__":
    base_directory = "eval"

    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)
        if os.path.isdir(folder_path):
            output_pickle = os.path.join(folder_path, f"{folder_name}_combined.pkl")
            combine_pickles_in_directory(folder_path, output_pickle)
