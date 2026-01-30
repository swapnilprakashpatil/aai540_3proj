import pandas as pd
from pathlib import Path
import argparse
import numpy as np

# Configuration
DATA_DIR = Path(__file__).parent  # Use script's directory
FULL_DATASET_URL = "https://download.cms.gov/openpayments/PGYR2024_P06302025_06162025/OP_DTL_GNRL_PGYR2024_P06302025_06162025.csv"
FULL_DATASET_LOCAL = "OP_DTL_GNRL_PGYR2024_P06302025_06162025.csv"
LIGHT_DATASET = "lightdataset.csv"
DEFAULT_TARGET_ROWS = 1000000
DEFAULT_CHUNK_SIZE = 100000  # Read 100k rows at a time
RANDOM_CHUNK_SIZE = 25000  # Chunk size for random selection

def create_light_dataset(use_url=True, random_sample=False, random_seed=42, target_rows=DEFAULT_TARGET_ROWS, chunk_size=DEFAULT_CHUNK_SIZE):

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
    
    print(f"Creating light dataset with {target_rows:,} records...")
    print(f"Source type: {source_type}")
    print(f"Source: {input_source}")
    print(f"Output: {output_file}")
    print(f"Chunk size: {chunk_size:,}")
    print(f"Random sampling: {'Enabled' if random_sample else 'Disabled'}")
    if random_sample:
        print(f"Random seed: {random_seed}")
        print(f"Random chunk size: {RANDOM_CHUNK_SIZE:,}")
    
    if use_url:
        print(f"\nDownloading data from CMS web in chunks...")
    else:
        print(f"\nReading local file in chunks...")
    
    # Set random seed for reproducibility
    if random_sample:
        np.random.seed(random_seed)
    
    rows_written = 0
    first_chunk = True
    
    if random_sample:
        # For random sampling, collect all data first in smaller chunks
        # Then randomly sample from the collected data
        print(f"\nCollecting data for random sampling...")
        all_data = []
        total_rows_read = 0
        
        for chunk_num, chunk in enumerate(pd.read_csv(input_source, chunksize=chunk_size, low_memory=False), 1):
            all_data.append(chunk)
            total_rows_read += len(chunk)
            print(f"  Chunk {chunk_num}: Read {len(chunk):,} rows (Total: {total_rows_read:,})")
        
        # Concatenate all chunks
        print(f"\nConcatenating {len(all_data)} chunks...")
        full_data = pd.concat(all_data, ignore_index=True)
        print(f"Total rows available: {len(full_data):,}")
        
        # Randomly sample target_rows
        sample_size = min(target_rows, len(full_data))
        print(f"\nRandomly sampling {sample_size:,} rows...")
        sampled_data = full_data.sample(n=sample_size, random_state=random_seed)
        
        # Write sampled data in chunks of RANDOM_CHUNK_SIZE
        print(f"\nWriting randomly sampled data in chunks of {RANDOM_CHUNK_SIZE:,}...")
        for i in range(0, len(sampled_data), RANDOM_CHUNK_SIZE):
            chunk_to_write = sampled_data.iloc[i:i+RANDOM_CHUNK_SIZE]
            
            if first_chunk:
                chunk_to_write.to_csv(output_file, index=False, mode='w')
                first_chunk = False
            else:
                chunk_to_write.to_csv(output_file, index=False, mode='a', header=False)
            
            rows_written += len(chunk_to_write)
            print(f"  Wrote chunk: {len(chunk_to_write):,} rows (Total: {rows_written:,})")
    else:
        # Sequential sampling (original behavior)
        for chunk_num, chunk in enumerate(pd.read_csv(input_source, chunksize=chunk_size, low_memory=False), 1):
            # Calculate how many rows to write from this chunk
            rows_to_write = min(len(chunk), target_rows - rows_written)
            
            # Write chunk (or part of it) to output file
            if first_chunk:
                chunk.iloc[:rows_to_write].to_csv(output_file, index=False, mode='w')
                first_chunk = False
            else:
                chunk.iloc[:rows_to_write].to_csv(output_file, index=False, mode='a', header=False)
            
            rows_written += rows_to_write
            print(f"  Chunk {chunk_num}: Wrote {rows_to_write:,} rows (Total: {rows_written:,})")
            
            # Stop if we've written enough rows
            if rows_written >= target_rows:
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
  
  # Random sampling from web
  python create_light_dataset.py --random
  
  # Random sampling from local file with custom seed
  python create_light_dataset.py --local --random --seed 123
  
  # Create smaller dataset with 500K rows
  python create_light_dataset.py --target-rows 500000
  
  # Use larger chunk size for faster processing
  python create_light_dataset.py --chunk-size 200000
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
    
    parser.add_argument(
        '--random',
        action='store_true',
        help='Randomly sample rows instead of taking sequential rows'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--target-rows',
        type=int,
        default=DEFAULT_TARGET_ROWS,
        help=f'Number of rows to include in the light dataset (default: {DEFAULT_TARGET_ROWS:,})'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f'Chunk size for reading data (default: {DEFAULT_CHUNK_SIZE:,})'
    )
    
    args = parser.parse_args()
    
    # Determine use_url based on arguments
    if args.local:
        create_light_dataset(
            use_url=False, 
            random_sample=args.random, 
            random_seed=args.seed,
            target_rows=args.target_rows,
            chunk_size=args.chunk_size
        )
    else:
        # Default: download from web
        create_light_dataset(
            use_url=True, 
            random_sample=args.random, 
            random_seed=args.seed,
            target_rows=args.target_rows,
            chunk_size=args.chunk_size
        )
