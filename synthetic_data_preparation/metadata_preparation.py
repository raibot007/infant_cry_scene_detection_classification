def read_file(file_path):
    """Reads the input file and returns rows as a list of lists, handling tab-separated values.
    Only considers the first two columns, ignoring additional columns if present."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip().split('\t')[:2] for line in lines]  # Split by tab and keep only first 2 columns

def write_file(file_path, data):
    """Writes the distinct rows to the file, ensuring tab-separated values."""
    with open(file_path, 'w') as f:
        for row in data:
            f.write('\t'.join(row) + '\n')  # Join by tab

def remove_duplicates_preserve_order(rows):
    """Removes duplicate rows from the list of rows while preserving the original order."""
    seen = set()
    distinct_rows = []
    for row in rows:
        row_tuple = tuple(row)  # Convert row to tuple for immutability and set operations
        if row_tuple not in seen:
            seen.add(row_tuple)
            distinct_rows.append(row)
    return distinct_rows

def process_file(input_file, output_file):
    """Processes the file to remove duplicate rows and writes the result to a new file, preserving order."""
    rows = read_file(input_file)
    distinct_rows = remove_duplicates_preserve_order(rows)
    write_file(output_file, distinct_rows)


# Example usage:
input_file = '/iitjfs/home/2023ree1048/sed_cry/meta/evaluation_setup/street_fold1_evaluate.txt'
output_file = '/iitjfs/home/2023ree1048/sed_cry/meta/evaluation_setup/street_fold1_test.txt'
process_file(input_file, output_file)

input_file = '/iitjfs/home/2023ree1048/sed_cry/meta/evaluation_setup/street_fold2_evaluate.txt'
output_file = '/iitjfs/home/2023ree1048/sed_cry/meta/evaluation_setup/street_fold2_test.txt'
process_file(input_file, output_file)

input_file = '/iitjfs/home/2023ree1048/sed_cry/meta/evaluation_setup/street_fold3_evaluate.txt'
output_file = '/iitjfs/home/2023ree1048/sed_cry/meta/evaluation_setup/street_fold3_test.txt'
process_file(input_file, output_file)

input_file = '/iitjfs/home/2023ree1048/sed_cry/meta/evaluation_setup/street_fold4_evaluate.txt'
output_file = '/iitjfs/home/2023ree1048/sed_cry/meta/evaluation_setup/street_fold4_test.txt'
process_file(input_file, output_file)

