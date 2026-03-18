import zipfile
import os
import time
from multiprocessing import Pool, Manager, cpu_count
import math
from pathlib import Path

def add_file_chunk(args):
    """Add a chunk of files to a separate zip archive in a single process"""
    file_chunk, base_path, zip_path, progress_dict, process_id = args
    
    added_count = 0
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
            for file_path in file_chunk:
                try:
                    # Calculate the archive name (relative path)
                    arcname = os.path.relpath(file_path, base_path)
                    zip_ref.write(file_path, arcname)
                    added_count += 1
                    
                    # Update progress every 50 files per process
                    if added_count % 50 == 0:
                        progress_dict[process_id] = added_count
                        
                except Exception as e:
                    print(f"Error adding {file_path}: {e}")
                    
        # Final update
        progress_dict[process_id] = added_count
        return added_count, zip_path
        
    except Exception as e:
        print(f"Process {process_id} error: {e}")
        return added_count, zip_path

def merge_zip_files(partial_zips, final_zip_path):
    """Merge multiple zip files into one final zip file"""
    print(f"Merging {len(partial_zips)} partial zip files...")
    
    with zipfile.ZipFile(final_zip_path, 'w', zipfile.ZIP_DEFLATED) as final_zip:
        for partial_zip in partial_zips:
            try:
                with zipfile.ZipFile(partial_zip, 'r') as partial:
                    for file_info in partial.filelist:
                        file_data = partial.read(file_info.filename)
                        final_zip.writestr(file_info, file_data)
                
                # Remove partial zip after merging
                os.remove(partial_zip)
                
            except Exception as e:
                print(f"Error merging {partial_zip}: {e}")
    
    print(f"Merge complete: {final_zip_path}")

def get_all_files(source_dir):
    """Recursively get all files in a directory"""
    file_list = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def zip_directory(source_dir, output_zip_path, max_workers=None):
    """
    Zip a directory using multiprocessing for improved performance.
    
    Args:
        source_dir: Directory to zip
        output_zip_path: Path for the output zip file
        max_workers: Number of worker processes (default: CPU count)
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist!")
        return
    
    # Determine optimal number of workers for HPC
    if max_workers is None:
        max_workers = cpu_count()
    
    print(f"Using {max_workers} processes for compression...")
    
    # Get all files to zip
    print(f"Scanning directory: {source_dir}")
    file_list = get_all_files(source_dir)
    total_files = len(file_list)
    
    if total_files == 0:
        print("No files found to zip!")
        return
    
    print(f"Found {total_files} files to compress...")
    
    # Split files into chunks for each process
    chunk_size = math.ceil(total_files / max_workers)
    file_chunks = [file_list[i:i + chunk_size] for i in range(0, total_files, chunk_size)]
    
    # Create shared progress tracking
    manager = Manager()
    progress_dict = manager.dict()
    
    # Create temporary directory for partial zips
    temp_dir = os.path.join(os.path.dirname(output_zip_path), '.temp_zips')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Prepare arguments for each process
    process_args = []
    for i, chunk in enumerate(file_chunks):
        if chunk:  # Only add non-empty chunks
            progress_dict[i] = 0
            partial_zip_path = os.path.join(temp_dir, f'partial_{i}.zip')
            process_args.append((chunk, source_dir, partial_zip_path, progress_dict, i))
    
    start_time = time.time()
    
    # Start multiprocessing
    print("Starting compression...")
    with Pool(processes=len(process_args)) as pool:
        # Start async processes
        results = pool.map_async(add_file_chunk, process_args)
        
        # Monitor progress
        while not results.ready():
            time.sleep(2)  # Check every 2 seconds
            total_completed = sum(progress_dict.values())
            
            if total_completed > 0:
                elapsed_time = time.time() - start_time
                estimated_total_time = (elapsed_time / total_completed) * total_files
                remaining_time = estimated_total_time - elapsed_time
                
                print(f"Compressed {total_completed}/{total_files} files. "
                      f"Estimated time remaining: {remaining_time:.2f} seconds")
        
        # Get final results
        compression_results = results.get()
    
    compression_time = time.time() - start_time
    total_compressed = sum([count for count, _ in compression_results])
    partial_zips = [zip_path for _, zip_path in compression_results]
    
    print(f"\nCompression phase complete: {compression_time:.2f} seconds")
    print(f"Total files compressed: {total_compressed}/{total_files}")
    
    # Merge all partial zips into final zip
    merge_start = time.time()
    merge_zip_files(partial_zips, output_zip_path)
    merge_time = time.time() - merge_start
    
    # Clean up temp directory
    try:
        os.rmdir(temp_dir)
    except:
        pass
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Zipping complete: {output_zip_path}")
    print(f"Compression time: {compression_time:.2f} seconds")
    print(f"Merge time: {merge_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total files: {total_compressed}/{total_files}")
    print(f"Average files per second: {total_compressed/total_time:.2f}")
    
    # Show file size
    if os.path.exists(output_zip_path):
        zip_size_mb = os.path.getsize(output_zip_path) / (1024 * 1024)
        print(f"Output zip size: {zip_size_mb:.2f} MB")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Example usage
    source_dir = 'Data/2025_cgras/segmentation_outputs/2024_genera_model_filtered_split_tiled_balanced'  # Replace with your source directory
    output_zip_path = 'Data/cgras_2025.zip'  # Replace with your desired output zip file path
    
    # For HPC, you can specify max_workers based on your system
    # e.g., zip_directory(source_dir, output_zip_path, max_workers=16)
    zip_directory(source_dir, output_zip_path)
