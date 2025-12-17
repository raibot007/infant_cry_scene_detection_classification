import random
import os

def read_file(file_path):
    """Reads the input file and returns rows as a list of lists, handling tab-separated values."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip().split('\t') for line in lines]  # Split by tab

def write_file(file_path, data):
    """Writes the data to the file, ensuring tab-separated values."""
    with open(file_path, 'w') as f:
        for row in data:
            f.write('\t'.join(row) + '\n')  # Join by tab

def split_filenames(rows, split_ratio=0.75):
    """Splits the data based on filenames into two parts."""
    # Extract unique filenames
    filenames = list(set([row[0] for row in rows]))
    random.shuffle(filenames)

    # Split filenames into two sets: 75% and 25%
    split_idx = int(len(filenames) * split_ratio)
    return filenames[:split_idx], filenames[split_idx:]

def filter_rows_by_filenames(rows, filenames):
    """Filters the rows that match the given filenames."""
    return [row for row in rows if row[0] in filenames]

def create_split_sets(input_file, output_dir, num_sets=4):
    """Creates 4 sets of files where each set has 75% and 25% split."""
    rows = read_file(input_file)

    for i in range(1, num_sets + 1):
        # Split filenames
        filenames_75, filenames_25 = split_filenames(rows)

        # Filter rows by filenames
        rows_75 = filter_rows_by_filenames(rows, filenames_75)
        rows_25 = filter_rows_by_filenames(rows, filenames_25)

        # Create directories for the set
        os.makedirs(output_dir, exist_ok=True)

        # Write the 75% and 25% splits to files
        file_75 = os.path.join(output_dir, f'set_{i}_75.txt')
        file_25 = os.path.join(output_dir, f'set_{i}_25.txt')
        
        write_file(file_75, rows_75)
        write_file(file_25, rows_25)

# Example usage:
input_file = '/iitjfs/home/2023ree1048/sed_cry/audio/street/audio_metadata.txt'
output_dir = '/iitjfs/home/2023ree1048/sed_cry/meta/evaluation_setup'
create_split_sets(input_file, output_dir)

