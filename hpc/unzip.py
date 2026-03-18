import tarfile
import os
import sys
import time
import subprocess
from pathlib import Path

def extract_tar_system(tar_path, extract_to):
    """Fast extraction using system tar command (much faster than Python)"""
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    print(f"Extracting {tar_path} using system tar command...")
    print(f"Target directory: {extract_to}")
    
    start_time = time.time()
    
    try:
        # Use system tar with progress monitoring via pv if available
        # -x: extract, -f: file, -v: verbose, -C: change to directory
        result = subprocess.run(
            ['tar', '-xf', tar_path, '-C', extract_to, '--verbose'],
            capture_output=False,
            text=True,
            check=True
        )
        
        total_time = time.time() - start_time
        print(f"\nExtraction complete!")
        print(f"Total extraction time: {total_time:.2f} seconds")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during extraction: {e}")
        raise
    except FileNotFoundError:
        print("System 'tar' command not found. Falling back to Python extraction...")
        return extract_tar_python(tar_path, extract_to)

def extract_tar_python(tar_path, extract_to):
    """Optimized Python-based extraction with progress reporting"""
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    print(f"Opening tar archive: {tar_path}")
    
    start_time = time.time()
    
    try:
        with tarfile.open(tar_path, 'r', bufsize=1024*1024) as tar_ref:
            # Get list of members
            members = tar_ref.getmembers()
            total_files = len(members)
            
            print(f"Starting extraction of {total_files} files/directories...")
            
            extracted_count = 0
            
            # Extract all at once (faster than one by one)
            tar_ref.extractall(path=extract_to, members=members)
            extracted_count = total_files
            
            total_time = time.time() - start_time
            
            print(f"Extracted {tar_path} to {extract_to}")
            print(f"Total extraction time: {total_time:.2f} seconds")
            print(f"Total items extracted: {extracted_count}/{total_files}")
            if total_time > 0:
                print(f"Average items per second: {extracted_count/total_time:.2f}")
            
    except Exception as e:
        print(f"Error opening tar file: {e}")
        raise

def extract_tar(tar_path, extract_to, use_system=True):
    """
    Extract tar archive with automatic method selection
    
    Args:
        tar_path: Path to the tar file
        extract_to: Directory to extract to
        use_system: If True (default), use system tar command for speed
    """
    if use_system:
        return extract_tar_system(tar_path, extract_to)
    else:
        return extract_tar_python(tar_path, extract_to)

if __name__ == "__main__":
    tar_path = 'Data/cslics/2025_nov_spawn/1422724372929.tar'  # Replace with your .tar file path
    extract_to = 'Data/cslics/2025_nov_spawn'  # Replace with your extraction directory path
    
    # Use system tar command for maximum speed (recommended)
    extract_tar(tar_path, extract_to, use_system=True)
    
    # Or use Python extraction (slower but more portable):
    # extract_tar(tar_path, extract_to, use_system=False)