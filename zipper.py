import os
import py7zr

def zip_zarr_to_7z(zarr_path, output_path):
    """
    Compress a .zarr directory into a .7z archive using py7zr.

    Args:
        zarr_path (str): Path to the .zarr directory.
        output_path (str): Path to the output .7z file.
    """
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"The path {zarr_path} does not exist.")
    if not zarr_path.endswith(".zarr"):
        raise ValueError("The input path must point to a .zarr directory.")
    
    # Compress the .zarr directory
    try:
        with py7zr.SevenZipFile(output_path, 'w') as archive:
            archive.writeall(zarr_path, arcname=os.path.basename(zarr_path))
        print(f"Successfully compressed {zarr_path} to {output_path}")
    except Exception as e:
        print(f"Error during compression: {e}")
        raise

# Example usage
if __name__ == "__main__":
    zarr_directory = "/Users/roberto.delprete/Library/CloudStorage/OneDrive-ESA/Desktop/DATASETS/fire tasi/fire_dataset.zarr"
    output_archive = "/Users/roberto.delprete/Library/CloudStorage/OneDrive-ESA/Desktop/DATASETS/fire tasi/fire_dataset.7z"
    # Compress the .zarr directory
    zip_zarr_to_7z(zarr_directory, output_archive)
