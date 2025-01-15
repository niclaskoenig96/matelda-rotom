import os
import json
import csv
import re

def extract_original_value(original):
    # Use regex to capture text after "VAL"
    match = re.search(r'VAL(.*)', original)
    return match.group(1).strip() if match else ''

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

def jsonl_to_custom_csv(jsonl_file, csv_file, csv_template):
    # Read the single JSONL entry
    with open(jsonl_file, 'r') as f:
        data = json.loads(f.readline().strip())
    
    sid = data.get('sid')
    original = data.get('original')
    augment_sentences = data.get('augment', [])

    # Extract the value after "VAL" from the original field
    original_value = extract_original_value(original)
    
    # Extract header name from the original field
    header_name = extract_header_name(original)
    
    # Process each augment sentence
    cleaned_augments = []
    for augment in augment_sentences:
        # Clean and split the augment by multiple spaces
        cleaned_parts = clean_and_split_augment(augment, header_name)
        cleaned_augments.extend(cleaned_parts)
    
    # Remove duplicates
    cleaned_augments = list(set(cleaned_augments))
    
    # Split the CSV template to get headers and initial row format
    headers = csv_template.strip().split(",")
    row_template = csv_template.replace('{sid}', str(sid))  # Replace any static values like SID in template

    # Check if the file exists and write headers only if starting fresh
    write_headers = not os.path.exists(csv_file)
    
    # Append to the CSV file
    with open(csv_file, 'a', newline='', encoding='utf-8') as csv_out:
        writer = csv.writer(csv_out)
        
        if write_headers:
            writer.writerow(headers)  # Write header row only once
        
        # Write each row, replacing only the {original} field with each augment
        for augment in cleaned_augments:
            # Skip the original value if it matches the augment
            if augment == original_value:
                continue
            
            # Replace {original} in the row template with the current augment
            row = row_template.replace('{original}', augment)
            # Write the row as a list to the CSV file
            writer.writerow(row.split(','))

