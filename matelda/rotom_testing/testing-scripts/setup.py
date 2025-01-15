import argparse
import pandas as pd
import subprocess
import os
import json
import jsonl_to_csv

def process_rows(temp_txt_file, full_row_csv_list, folder_path):
    # Define paths (update these paths as needed)
    table_name = os.path.basename(os.path.normpath(folder_path))
    model_path = f"invda/models/cleaning_{table_name}/"  
    jsonl_output_file = f"{temp_txt_file}.augment.jsonl"

    # Construct the shell command
    command = f"CUDA_VISIBLE_DEVICES=2 python invda/generate.py --input {temp_txt_file} --model_path {model_path} --type em"
    print(f"Processing input file: {temp_txt_file}")

    # Execute the shell command
    subprocess.run(command, shell=True)

    # Process the generated JSONL file
    if os.path.exists(jsonl_output_file) and os.path.getsize(jsonl_output_file) > 0:
        # Process each line in the JSONL file and map it to the corresponding full row CSV
        with open(jsonl_output_file, 'r') as jsonl_file:
            for index, line in enumerate(jsonl_file):
                jsonl_to_csv.jsonl_to_custom_csv(jsonl_output_file, f"{folder_path}/output.csv", full_row_csv_list[index])

    # Clean up temporary files
    #os.remove(temp_txt_file)
    # Uncomment the following if you want to clean up JSONL output as well
    # os.remove(jsonl_output_file)


def csv_to_function(input_csv):
    # Load the CSV data
    data = pd.read_csv(input_csv)
    folder_path = os.path.dirname(input_csv)

    # Create a single temporary text file
    temp_txt_file = os.path.join(folder_path, "train.txt")
    full_row_csv_list = []  # To store full row CSVs for mapping JSONL outputs
    
    # Open the text file for writing
    with open(temp_txt_file, 'w') as txt_file:
        # Iterate over each row
        for row_index, row in data.iterrows():
            # Get the column index from the last column of the row
            column_index = int(row.iloc[-2])  # Assuming the last column contains the column index
            label = int(row.iloc[-1])
            # Check if the column index is within the valid range
            if 0 <= column_index < len(data.columns):
                # Get the column name and value at the specified index
                column_name = data.columns[column_index]
                column_value = row[column_index]
                

                
                # Create the parsed string in the format: COL *columnname* VAL *value*
                
                parsed_string = f'COL {column_name} VAL {column_value} \t{label}'

                # Write the parsed string to the file
                txt_file.write(parsed_string + '\n')

                # Convert the full row to a list of values and replace the specified entry
                full_row_values = list(map(str, row.values))
                full_row_values[column_index] = "{original}"
                
                # Add the modified row to the list
                full_row_csv_list.append(','.join(full_row_values))
            else:
                # Handle the case where the column index is out of range
                error_message = f"Error: Column index '{column_index}' is out of range"
                txt_file.write(error_message + '\n')
                full_row_csv_list.append(','.join(map(str, row.values)))  # Use original row as fallback

    # Call the processing function with the combined file
    process_rows(temp_txt_file, full_row_csv_list, folder_path)


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process a CSV file and handle row transformations.")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    args = parser.parse_args()

    # Call the main function with the provided CSV file
    csv_to_function(args.input_csv)
