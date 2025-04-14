# To install the py7zr library, use the following command:
# pip install py7zr

import os
import py7zr
import argparse

def zip_zarr_to_7z(zarr_path, output_path, max_size_gb=10):
    """
    Compress a .zarr directory into a .7z archive using py7zr, with optional splitting.

    Args:
        zarr_path (str): Path to the .zarr directory.
        output_path (str): Path to the output .7z file.
        max_size_gb (int): Maximum size of each split archive in GB.
    """
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"The path {zarr_path} does not exist.")
    if not zarr_path.endswith(".zarr"):
        raise ValueError("The input path must point to a .zarr directory.")
    
    # Convert max size to bytes
    max_size_bytes = max_size_gb * 1024**3

    # Compress the .zarr directory with splitting
    try:
        with py7zr.SevenZipFile(output_path, 'w', volume=max_size_bytes) as archive:
            archive.writeall(zarr_path, arcname=os.path.basename(zarr_path))
        print(f"Successfully compressed {zarr_path} to {output_path} with max size {max_size_gb} GB per volume.")
    except Exception as e:
        print(f"Error during compression: {e}")
        raise

# Example usage with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress a .zarr directory into a .7z archive with optional splitting.")
    parser.add_argument("zarr_path", type=str, help="Path to the .zarr directory.")
    parser.add_argument("output_path", type=str, help="Path to the output .7z file.")
    parser.add_argument("--max_size_gb", type=int, default=10, help="Maximum size of each split archive in GB (default: 10).")

    args = parser.parse_args()

    # Compress the .zarr directory with the provided arguments
    zip_zarr_to_7z(args.zarr_path, args.output_path, max_size_gb=args.max_size_gb)
