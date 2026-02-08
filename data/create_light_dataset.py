import pandas as pd
from pathlib import Path
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import io

# Configuration
DATA_DIR = Path(__file__).parent  # Use script's directory
FULL_DATASET_URL = "https://download.cms.gov/openpayments/PGYR2024_P06302025_06162025/OP_DTL_GNRL_PGYR2024_P06302025_06162025.csv"
FULL_DATASET_LOCAL = "OP_DTL_GNRL_PGYR2024_P06302025_06162025.csv"
LIGHT_DATASET = "lightdataset1.csv"
DEFAULT_TARGET_ROWS = 1000000
DEFAULT_CHUNK_SIZE = 100000  # Read 100k rows at a time
RANDOM_CHUNK_SIZE = 25000  # Chunk size for random selection
DEFAULT_NUM_THREADS = 4  # Number of parallel threads for reading

def read_chunk_parallel(input_source, skiprows, nrows, chunk_id):
    """Read a single chunk of data in parallel."""
    try:
        # For skiprows > 0, we need to skip rows but keep the header
        if skiprows > 0:
            chunk = pd.read_csv(input_source, skiprows=range(1, skiprows + 1), nrows=nrows, low_memory=False)
        else:
            chunk = pd.read_csv(input_source, nrows=nrows, low_memory=False)
        return chunk_id, chunk, None
    except Exception as e:
        return chunk_id, None, str(e)

def create_light_dataset(use_url=True, random_sample=False, random_seed=42, target_rows=DEFAULT_TARGET_ROWS, chunk_size=DEFAULT_CHUNK_SIZE, num_threads=DEFAULT_NUM_THREADS):

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
    print(f"Number of threads: {num_threads}")
    print(f"Random sampling: {'Enabled' if random_sample else 'Disabled'}")
    if random_sample:
        print(f"Random seed: {random_seed}")
        print(f"Random chunk size: {RANDOM_CHUNK_SIZE:,}")
    
    if use_url:
        print(f"\nDownloading data from CMS web in parallel chunks...")
    else:
        print(f"\nReading local file in parallel chunks...")
    
    # Set random seed for reproducibility
    if random_sample:
        np.random.seed(random_seed)
    
    rows_written = 0
    first_chunk = True
    write_lock = threading.Lock()
    
    if random_sample:
        # For random sampling, collect all data first using parallel reading
        print(f"\nCollecting data for random sampling using {num_threads} threads...")
        all_data = []
        total_rows_read = 0
        
        # Calculate number of chunks needed
        num_chunks_to_read = (target_rows // chunk_size) + 2  # Read a bit extra
        
        # Create tasks for parallel reading
        chunk_tasks = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for chunk_id in range(num_chunks_to_read):
                skiprows = chunk_id * chunk_size
                future = executor.submit(read_chunk_parallel, input_source, skiprows, chunk_size, chunk_id)
                futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures):
                chunk_id, chunk, error = future.result()
                if error:
                    print(f"  Error reading chunk {chunk_id}: {error}")
                    continue
                if chunk is not None and len(chunk) > 0:
                    all_data.append(chunk)
                    total_rows_read += len(chunk)
                    print(f"  Thread completed chunk {chunk_id}: Read {len(chunk):,} rows (Total: {total_rows_read:,})")
                else:
                    # No more data available
                    break
        
        if not all_data:
            print("Error: No data collected")
            return
        
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
            
            with write_lock:
                if first_chunk:
                    chunk_to_write.to_csv(output_file, index=False, mode='w')
                    first_chunk = False
                else:
                    chunk_to_write.to_csv(output_file, index=False, mode='a', header=False)
            
            rows_written += len(chunk_to_write)
            print(f"  Wrote chunk: {len(chunk_to_write):,} rows (Total: {rows_written:,})")
    else:
        # Sequential sampling with parallel reading
        print(f"\nReading data using {num_threads} parallel threads...")
        
        # Calculate number of chunks needed
        num_chunks_needed = (target_rows + chunk_size - 1) // chunk_size
        
        # Use a queue to maintain order
        chunk_queue = Queue()
        chunks_dict = {}
        next_chunk_to_write = 0
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for chunk_id in range(num_chunks_needed):
                skiprows = chunk_id * chunk_size
                future = executor.submit(read_chunk_parallel, input_source, skiprows, chunk_size, chunk_id)
                futures.append((chunk_id, future))
            
            # Process results as they complete
            for chunk_id, future in futures:
                chunk_id_result, chunk, error = future.result()
                
                if error:
                    print(f"  Error reading chunk {chunk_id_result}: {error}")
                    break
                    
                if chunk is None or len(chunk) == 0:
                    print(f"  Chunk {chunk_id_result}: No more data available")
                    break
                
                print(f"  Thread completed chunk {chunk_id_result}: Read {len(chunk):,} rows")
                
                # Store chunk in dictionary
                chunks_dict[chunk_id_result] = chunk
                
                # Write all consecutive chunks that are ready
                while next_chunk_to_write in chunks_dict:
                    chunk_to_write = chunks_dict.pop(next_chunk_to_write)
                    
                    # Calculate how many rows to write from this chunk
                    rows_to_write = min(len(chunk_to_write), target_rows - rows_written)
                    
                    with write_lock:
                        if first_chunk:
                            chunk_to_write.iloc[:rows_to_write].to_csv(output_file, index=False, mode='w')
                            first_chunk = False
                        else:
                            chunk_to_write.iloc[:rows_to_write].to_csv(output_file, index=False, mode='a', header=False)
                    
                    rows_written += rows_to_write
                    print(f"  Wrote chunk {next_chunk_to_write}: {rows_to_write:,} rows (Total: {rows_written:,})")
                    next_chunk_to_write += 1
                    
                    # Stop if we've written enough rows
                    if rows_written >= target_rows:
                        break
                
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
  # Download from web (default) with 4 threads
  python create_light_dataset.py
  
  # Use local file with 8 threads
  python create_light_dataset.py --local --threads 8
  
  # Explicitly download from web with 2 threads
  python create_light_dataset.py --web --threads 2
  
  # Random sampling from web with 6 threads
  python create_light_dataset.py --random --threads 6
  
  # Random sampling from local file with custom seed
  python create_light_dataset.py --local --random --seed 123
  
  # Create smaller dataset with 500K rows
  python create_light_dataset.py --target-rows 500000
  
  # Use larger chunk size for faster processing
  python create_light_dataset.py --chunk-size 200000 --threads 8
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
    
    parser.add_argument(
        '--threads',
        type=int,
        default=DEFAULT_NUM_THREADS,
        help=f'Number of parallel threads for reading data (default: {DEFAULT_NUM_THREADS})'
    )
    
    args = parser.parse_args()
    
    # Determine use_url based on arguments
    if args.local:
        create_light_dataset(
            use_url=False, 
            random_sample=args.random, 
            random_seed=args.seed,
            target_rows=args.target_rows,
            chunk_size=args.chunk_size,
            num_threads=args.threads
        )
    else:
        # Default: download from web
        create_light_dataset(
            use_url=True, 
            random_sample=args.random, 
            random_seed=args.seed,
            target_rows=args.target_rows,
            chunk_size=args.chunk_size,
            num_threads=args.threads
        )
