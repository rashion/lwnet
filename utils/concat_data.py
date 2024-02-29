import pandas as pd
import argparse

# Initialize the parser
parser = argparse.ArgumentParser(description='Concatenate two CSV files.')

# Adding arguments
parser.add_argument('-i', '--input_files', nargs=2, help='Input CSV file paths', required=True)
parser.add_argument('-o', '--output_file', help='Output CSV file path', required=True)

# Parse the arguments
args = parser.parse_args()

# Read the CSV files
df1 = pd.read_csv(args.input_files[0])
df2 = pd.read_csv(args.input_files[1])

# Concatenate the DataFrames
concatenated_df = pd.concat([df1, df2], ignore_index=True)

# Save the concatenated DataFrame to the specified output CSV file
concatenated_df.to_csv(args.output_file, index=False)
