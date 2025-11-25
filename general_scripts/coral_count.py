#! /usr/bin/env python3

"""
Script to count the number of alive and dead corals in a dataset by using the label files.


"""
import os
import sys
import glob
from pathlib import Path
import argparse

def count_corals(folder_path, file_extension=".txt"):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"Error: The folder '{folder_path}' does not exist.")
        return None
    
    extension = f"*{file_extension}"
    label_files = list(folder_path.glob(extension))
    if not label_files:
        print(f"No files with extension '{file_extension}' found in '{folder_path}'.")
        return None
    
    # Initialize counters
    results = {
        'dead_coral_total': 0,   # 0s = dead coral
        'alive_coral_total': 0   # 1s = alive coral
    }

    for file_path in label_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
            
            # Skip empty files but continue processing other files
            if not content:
                continue
            
            elements = content.split()
            for element in elements:
                try:
                    # Convert to float first, then to int to handle "0.0", "1.0" etc.
                    
                    if element == '0':
                        results['alive_coral_total'] += 1
                    elif element == '1':
                        results['dead_coral_total'] += 1
                    # Ignore all other numbers (2, 3, decimals, etc.)
                except ValueError:
                    # Skip non-numeric elements
                    continue
            
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue
    
    return results
        
def print_results(results):
    if results is None:
        print("No valid data found.")
        return
    
    print(f"Total alive corals (0s): {results['alive_coral_total']}")
    print(f"Total dead corals (1s): {results['dead_coral_total']}")
    total = results['dead_coral_total'] + results['alive_coral_total']
    print(f"Total coral instances: {total}")
    
    if total == 0:
        print("No corals found in the dataset.")
    else:
        alive_percent = (results['alive_coral_total'] / total) * 100
        dead_percent = (results['dead_coral_total'] / total) * 100
        print(f"Alive: {alive_percent:.1f}%, Dead: {dead_percent:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Count alive/dead corals from label files')
    parser.add_argument('folder', help='Path to folder containing text files')
    parser.add_argument('--ext', default='.txt', help='File extension to analyze (default: .txt)')
    
    args = parser.parse_args()
    
    results = count_corals(args.folder, args.ext)
    print_results(results)

if __name__ == "__main__":
    main()
