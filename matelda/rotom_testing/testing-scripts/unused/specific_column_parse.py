import pandas as pd


def csv_to_index_specific_txt(input_csv, output_txt):
    # Load the CSV data
    data = pd.read_csv(input_csv)

    # Open the output file
    with open(output_txt, 'w') as txt_file:
        # Iterate over each row
        for _, row in data.iterrows():
            # Get the column index from the last column of the row
            column_index = int(row.iloc[-1])  # Assuming the last column contains the column index

            # Check if the column index is within the valid range
            if 0 <= column_index < len(data.columns):
                # Get the column name and value at the specified index
                column_name = data.columns[column_index]
                column_value = row[column_index]

                # Write the output in the specified format
                txt_file.write(f'COL {column_name} VAL {column_value} \n')
            else:
                # Handle the case where the column index is out of range
                txt_file.write(f"Error: Column index '{column_index}' is out of range\n")

    print(f"Data has been parsed and saved to {output_txt}")


# Example usage
input_csv = 'test.csv'  # Replace with the path to your CSV file
output_txt = 'specific_column_parse.txt'
csv_to_index_specific_txt(input_csv, output_txt)
