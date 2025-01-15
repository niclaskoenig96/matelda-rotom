import csv
import pandas as pd

import csv

import csv
import random

import csv
import random

def process_entries(entries, output_csv):
    # Prepare a list to store processed rows
    processed_rows = []
    headers = None

    for entry in entries:
        csv_path, clean_path, row_number, column_index = entry

        with open(csv_path, newline='') as csvfile, open(clean_path, newline='') as cleanfile:
            reader = csv.reader(csvfile)
            clean_reader = csv.reader(cleanfile)

            # Read headers
            current_headers = next(reader)

            if headers is None:
                # Initialize headers with "Selector" and "Label" added
                headers = current_headers + ["Selector"] + ["Label"]

            # Read through rows
            csv_row = None
            clean_row = None

            for i, (csv_row_data, clean_row_data) in enumerate(zip(reader, clean_reader), start=1):
                if i == row_number:
                    csv_row = csv_row_data
                    clean_row = clean_row_data
                    break

            if csv_row and clean_row:
                # Get values to compare
                csv_value = csv_row[column_index]
                clean_value = clean_row[column_index]
                label = 0 if csv_value == clean_value else 1

                # Append processed row
                selector_value = column_index
                processed_row = csv_row + [selector_value] + [label]
                processed_rows.append(processed_row)

    # Ensure balance by adding more 0s and 1s if needed
    count_0 = sum(1 for row in processed_rows if row[-1] == 0)
    count_1 = sum(1 for row in processed_rows if row[-1] == 1)

    with open(entries[0][0], newline='') as csvfile, open(entries[0][1], newline='') as cleanfile:
        reader = list(csv.reader(csvfile))
        clean_reader = list(csv.reader(cleanfile))

        while count_0 < count_1:
            random_row = random.randint(1, len(reader) - 1)
            random_col = random.randint(0, len(reader[0]) - 1)

            csv_value = reader[random_row][random_col]
            clean_value = clean_reader[random_row][random_col]
            label = 0 if csv_value == clean_value else 1

            if label == 0:
                selector_value = random_col
                processed_row = reader[random_row] + [selector_value] + [label]
                processed_rows.append(processed_row)
                count_0 += 1

        while count_1 < count_0:
            random_row = random.randint(1, len(reader) - 1)
            random_col = random.randint(0, len(reader[0]) - 1)

            csv_value = reader[random_row][random_col]
            clean_value = clean_reader[random_row][random_col]
            label = 0 if csv_value == clean_value else 1

            if label == 1:
                selector_value = random_col
                processed_row = reader[random_row] + [selector_value] + [label]
                processed_rows.append(processed_row)
                count_1 += 1

    # Write the processed rows to a new CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(processed_rows)


def csv_to_formatted_txt(input_csv, output_txt):
    # Load the CSV data
    data = pd.read_csv(input_csv)

    # Open the output file
    with open(output_txt, 'w') as txt_file:
        # Iterate over each row and column
        for _, row in data.iterrows():
            for column in data.columns:
                # Write in the specified format: COL *columnname* VAL *value*
                txt_file.write(f'COL {column} VAL {row[column]} \t0\n')

    

