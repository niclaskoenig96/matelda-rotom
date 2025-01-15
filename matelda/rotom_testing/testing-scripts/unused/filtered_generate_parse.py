import json
import csv
import re
from collections import defaultdict


def extract_header_name(original):
    # Use regex to capture text between "COL" and "VAL"
    match = re.search(r'COL(.*?)VAL', original)
    return match.group(1).strip() if match else ''


def clean_and_split_augment(augment, header_name):
    # Remove "COL", "VAL", and the header name
    cleaned = augment.replace("COL", "").replace("VAL", "").replace(header_name, "").strip()
    # Split on two or more spaces
    parts = re.split(r'\s{2,}', cleaned)
    # Return non-empty parts
    return [part.strip() for part in parts if part.strip()]


def jsonl_to_csv(jsonl_file, csv_file):
    # Dictionary to store all cleaned, unique augment parts for each (sid, extracted header) combination
    data_dict = defaultdict(lambda: defaultdict(set))

    # Set to keep track of all unique headers extracted from original strings
    all_headers = set()

    # First pass: populate the data dictionary and extract headers
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            sid = data.get('sid')
            original = data.get('original')
            augment_sentences = data.get('augment', [])

            # Extract header name from the original field
            header_name = extract_header_name(original)
            all_headers.add(header_name)

            # Process each augment sentence
            for augment in augment_sentences:
                # Clean and split the augment by multiple spaces
                cleaned_parts = clean_and_split_augment(augment, header_name)
                for part in cleaned_parts:
                    # Add each cleaned part to a set to ensure uniqueness
                    data_dict[sid][header_name].add(part)

    # Prepare the headers
    headers = ['sid'] + sorted(all_headers)

    # Write to CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as csv_out:
        writer = csv.writer(csv_out)
        writer.writerow(headers)  # Write header row

        # Get the maximum number of augment parts across all (sid, header) pairs
        max_rows = max(len(parts) for sid_data in data_dict.values() for parts in sid_data.values())

        # Write each row for the maximum length
        for sid, headers_dict in data_dict.items():
            for row_idx in range(max_rows):
                row = [sid]  # Start each row with the sid
                # Append each unique cleaned augment part for the corresponding header
                row_data = []
                for header in headers[1:]:  # Skip 'sid' in headers
                    # Get the augment part if available; otherwise, add an empty string
                    augments = list(headers_dict[header])
                    row_data.append(augments[row_idx] if row_idx < len(augments) else '')

                # Skip row if all fields except 'sid' are empty
                if any(row_data):
                    writer.writerow([sid] + row_data)


# Usage
jsonl_to_csv('input.jsonl', 'output.csv')
