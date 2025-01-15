import json
import csv
import pandas as pd


def csv_to_formatted_txt(input_csv, output_txt):
    # Load the CSV data
    data = pd.read_csv(input_csv)

    # Open the output file
    with open(output_txt, 'w') as txt_file:
        # Iterate over each row and column
        for _, row in data.iterrows():
            for column in data.columns:
                # Write in the specified format: COL *columnname* VAL *value*
                txt_file.write(f'COL {column} VAL {row[column]} \t 0\n')

    print(f"Data has been formatted and saved to {output_txt}")



csv_to_formatted_txt('clean.csv', 'output.txt')
