import mdtraj as md
import numpy as np
import sys
import os
import re

def extract_frames(xtc_path, gro_path, cluster_indices_file, output_prefix):
    # Load the trajectory
    traj = md.load(xtc_path, top=gro_path)

    # Read cluster indices from the file
    try:
        with open(cluster_indices_file, 'r') as f:
            cluster_lines = f.readlines()
    except Exception as e:
        print(f"Error reading the cluster indices file: {e}")
        sys.exit(1)

    # Extract frames based on the provided cluster indices
    for cluster_line in cluster_lines:
        match = re.match(r'Cluster (\d+): (\[.+\])', cluster_line)
        if match:
            cluster_num = match.group(1)
            frame_indices = np.array(eval(match.group(2)), dtype=int)

            # Extract frames for the current cluster
            extracted_traj = traj[frame_indices]

            # Save each extracted frame to a separate .gro file with the specified prefix and cluster number
            for i, frame_index in enumerate(frame_indices):
                output_filename = f"{output_prefix}_Cluster{cluster_num}_Frame{frame_index}.gro"
                extracted_traj[i].save_gro(output_filename)

if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 5:
        print("Usage: python extract_highdens.py <xtc_file> <gro_file> <cluster_indices_file> <output_prefix>")
        sys.exit(1)

    xtc_file, gro_file, cluster_indices_file, output_prefix = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    # Extract frames and save to separate .gro files with the specified prefix and cluster number
    extract_frames(xtc_file, gro_file, cluster_indices_file, output_prefix)

