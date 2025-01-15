import argparse
import pandas as pd
import subprocess
import os
import json
import jsonl_to_csv

def process_row(parsed_string, full_row_csv, row_index, folder_path):
    # Write parsed_string to a temporary file
    temp_txt_file = f"temp_row_{row_index}.txt"
    with open(temp_txt_file, 'w') as txt_file:
        txt_file.write(parsed_string)

    # Define paths (update these paths as needed)
    table_name = os.path.basename(os.path.normpath(folder_path))
    model_path = f"invda/models/cleaning_{table_name}/"  
    jsonl_output_file = f"{temp_txt_file}.augment.jsonl"


    # Construct the shell command
    command = f"CUDA_VISIBLE_DEVICES=2 python invda/generate.py --input {temp_txt_file} --model_path {model_path} --type em"
    print(parsed_string)
    # Execute the shell command
    subprocess.run(command, shell=True)

    # Process the generated JSONL file
    if os.path.exists(jsonl_output_file) and os.path.getsize(jsonl_output_file) > 0:
        jsonl_to_csv.jsonl_to_custom_csv(jsonl_output_file, f"{folder_path}/output.csv", full_row_csv)



    # Clean up temporary files
    os.remove(temp_txt_file)
    #if os.path.exists(jsonl_output_file):
        #os.remove(jsonl_output_file)

def csv_to_function(input_csv):
    # Load the CSV data
    data = pd.read_csv(input_csv)
    folder_path = os.path.dirname(input_csv)

    # Iterate over each row
    for row_index, row in data.iterrows():
        # Get the column index from the last column of the row
        column_index = int(row.iloc[-1])  # Assuming the last column contains the column index

        # Check if the column index is within the valid range
        if 0 <= column_index < len(data.columns):
            # Get the column name and value at the specified index
            column_name = data.columns[column_index]
            column_value = row[column_index]
            
            # Create the parsed string in the format: COL *columnname* VAL *value*
            parsed_string = f'COL {column_name} VAL {column_value} \t 0'
            
            # Convert the full row to a list of values and replace the specified entry
            full_row_values = list(map(str, row.values))
            full_row_values[column_index] = "{original}"
            
            # Join the modified row back into a CSV format
            full_row_csv = ','.join(full_row_values)
            
            # Call the processing function
            process_row(parsed_string, full_row_csv, row_index, folder_path)
        else:
            # Handle the case where the column index is out of range
            error_message = f"Error: Column index '{column_index}' is out of range"
            full_row_csv = ','.join(map(str, row.values))  # Original row as fallback
            process_row(error_message, full_row_csv, row_index, folder_path)



if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process a CSV file and handle row transformations.")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    args = parser.parse_args()

    # Call the main function with the provided CSV file
    csv_to_function(args.input_csv)

