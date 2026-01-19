import pandas as pd
from pathlib import Path

# Configuration
DATA_DIR = Path("data")
FULL_DATASET = "OP_DTL_GNRL_PGYR2024_P06302025_06162025.csv"
LIGHT_DATASET = "lightdataset.csv"
TARGET_ROWS = 1000000
CHUNK_SIZE = 100000  # Read 100k rows at a time

def create_light_dataset():

    input_file = DATA_DIR / FULL_DATASET
    output_file = DATA_DIR / LIGHT_DATASET
    
    print(f"Creating light dataset with {TARGET_ROWS:,} records...")
    print(f"Source: {input_file}")
    print(f"Output: {output_file}")
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    rows_written = 0
    first_chunk = True
    
    # Read in chunks
    print(f"\nReading in chunks of {CHUNK_SIZE:,} rows...")
    
    for chunk_num, chunk in enumerate(pd.read_csv(input_file, chunksize=CHUNK_SIZE, low_memory=False), 1):
        # Calculate how many rows to write from this chunk
        rows_to_write = min(len(chunk), TARGET_ROWS - rows_written)
        
        # Write chunk (or part of it) to output file
        if first_chunk:
            chunk.iloc[:rows_to_write].to_csv(output_file, index=False, mode='w')
            first_chunk = False
        else:
            chunk.iloc[:rows_to_write].to_csv(output_file, index=False, mode='a', header=False)
        
        rows_written += rows_to_write
        print(f"  Chunk {chunk_num}: Wrote {rows_to_write:,} rows (Total: {rows_written:,})")
        
        # Stop if we've written enough rows
        if rows_written >= TARGET_ROWS:
            break
    
    print(f"\nLight dataset created successfully!")
    print(f"  Total rows: {rows_written:,}")
    print(f"  Output file: {output_file}")
    
    # Display file size
    file_size_mb = output_file.stat().st_size / (1024 ** 2)
    print(f"  File size: {file_size_mb:.2f} MB")
    
    # Show sample of data
    print("\nSample of light dataset (first 5 rows):")
    sample = pd.read_csv(output_file, nrows=5)
    print(f"  Columns: {len(sample.columns)}")
    print(f"  Sample:\n{sample.head()}")

if __name__ == "__main__":
    create_light_dataset()
