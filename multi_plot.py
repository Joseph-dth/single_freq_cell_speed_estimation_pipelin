
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import re

def plot_speed_analysis(folder_path):
    """
    Analyzes cell speed data from CSV files in a specified folder and generates plots.

    This function searches for files ending with '_speeds_filtered.csv' in the given
    folder. It extracts frequency information from the filenames and reads the speed
    data from the 'speed_um_per_s' column in each CSV.

    It then generates two plots:
    1. A scatter plot of frequency vs. speed for every data point.
    2. A line plot showing the average speed for each frequency, with error bars
       representing the standard deviation.

    Args:
        folder_path (str): The absolute path to the folder containing the CSV files.

    Returns:
        None. The plots are displayed directly.
    """
    all_dfs = []
    for filename in os.listdir(folder_path):
        if filename.endswith("_speeds_filtered.csv"):
            try:
                # Extract frequency from filename using regex
                match = re.match(r"ipsc_(\d+[MKmk]?)_", filename, re.IGNORECASE)
                if not match:
                    print(f"Warning: Could not extract frequency from '{filename}'. Skipping.")
                    continue
                
                freq_str = match.group(1)
                if 'M' in freq_str:
                    frequency = float(freq_str.replace('M', '')) * 1e6
                elif 'k' in freq_str:
                    frequency = float(freq_str.replace('k', '')) * 1e3
                else:
                    frequency = float(freq_str)

                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                df['frequency'] = frequency
                all_dfs.append(df)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    if not all_dfs:
        print("No valid data files found. Exiting.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Plot 1: Scatter plot of all data points
    plt.figure(figsize=(10, 6))
    plt.scatter(combined_df['frequency'], combined_df['speed_um_per_s'], alpha=0.5)
    plt.xscale('log')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Speed (µm/s)")
    plt.title("Frequency vs. Speed (All Data Points)")
    plt.grid(True)
    plt.show()

    # Plot 2: Average curve with standard deviation
    plt.figure(figsize=(10, 6))
    avg_speeds = combined_df.groupby('frequency')['speed_um_per_s'].mean()
    std_speeds = combined_df.groupby('frequency')['speed_um_per_s'].std()
    
    # Sort by frequency for a clean plot
    avg_speeds = avg_speeds.sort_index()
    std_speeds = std_speeds.sort_index()

    plt.errorbar(avg_speeds.index, avg_speeds.values, yerr=std_speeds.values, 
                 fmt='-o', capsize=5, label='Average Speed ± Std Dev')
    plt.xscale('log')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Average Speed (µm/s)")
    plt.title("Average Speed vs. Frequency with Standard Deviation")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot speed analysis from CSV files in a folder.")
    parser.add_argument("folder_path", type=str, help="The absolute path to the folder containing the data files.")
    args = parser.parse_args()
    
    plot_speed_analysis(args.folder_path)
