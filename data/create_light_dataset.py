import pandas as pd
from pathlib import Path
import argparse

# Configuration
DATA_DIR = Path("data")
FULL_DATASET_URL = "https://download.cms.gov/openpayments/PGYR2024_P06302025_06162025/OP_DTL_GNRL_PGYR2024_P06302025_06162025.csv"
FULL_DATASET_LOCAL = "OP_DTL_GNRL_PGYR2024_P06302025_06162025.csv"
LIGHT_DATASET = "lightdataset.csv"
TARGET_ROWS = 1000000
CHUNK_SIZE = 100000  # Read 100k rows at a time

def create_light_dataset(use_url=True):

    # Determine input source
    if use_url:
        input_source = FULL_DATASET_URL
        source_type = "Web"
    else:
        input_source = str(DATA_DIR / FULL_DATASET_LOCAL)
        source_type = "Local file"
        # Check if local file exists
        if not Path(input_source).exists():
            print(f"Error: Local file not found: {input_source}")
            print("Tip: Use --web flag to download from CMS instead.")
            return
    
    output_file = DATA_DIR / LIGHT_DATASET
    
    print(f"Creating light dataset with {TARGET_ROWS:,} records...")
    print(f"Source type: {source_type}")
    print(f"Source: {input_source}")
    print(f"Output: {output_file}")
    
    if use_url:
        print(f"\nDownloading data from CMS web in chunks...")
    else:
        print(f"\nReading local file in chunks...")
    
    rows_written = 0
    first_chunk = True
    
    for chunk_num, chunk in enumerate(pd.read_csv(input_source, chunksize=CHUNK_SIZE, low_memory=False), 1):
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
    parser = argparse.ArgumentParser(
        description="Create a light dataset from CMS Open Payments data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from web (default)
  python create_light_dataset.py
  
  # Use local file
  python create_light_dataset.py --local
  
  # Explicitly download from web
  python create_light_dataset.py --web
        """
    )
    
    parser.add_argument(
        '--local',
        action='store_true',
        help='Use local file instead of downloading from URL'
    )
    
    parser.add_argument(
        '--web',
        action='store_true',
        help='Download from web (default behavior)'
    )
    
    args = parser.parse_args()
    
    # Determine use_url based on arguments
    if args.local:
        create_light_dataset(use_url=False)
    else:
        # Default: download from web
        create_light_dataset(use_url=True)
