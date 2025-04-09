import os
import logging
from huggingface_hub import HfApi
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def upload_file_to_huggingface(file_path: str, repo_id: str, repo_type: str = "dataset") -> None:
    """Uploads a single file to the Hugging Face Hub.

    Args:
        file_path (str): Path to the local file.
        repo_id (str): Hugging Face repository ID.
        repo_type (str, optional): Repository type (default is "dataset").

    Raises:
        Exception: If an error occurs during upload.
    """
    api = HfApi()
    try:
        file_name = os.path.basename(file_path)
        logging.info(f"Uploading file '{file_name}' to Hugging Face repo '{repo_id}' ({repo_type})...")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_name,
            repo_id=repo_id,
            repo_type=repo_type
        )
        logging.info("File upload completed successfully.")
    except Exception as e:
        logging.error(f"File upload failed: {e}")
        raise e


def upload_to_huggingface(folder_path: str, repo_id: str, repo_type: str = "dataset") -> None:
    """Uploads a local folder to the Hugging Face Hub.

    Args:
        folder_path (str): Path to the local folder.
        repo_id (str): Hugging Face repository ID.
        repo_type (str, optional): Repository type (default is "dataset").

    Raises:
        Exception: If an error occurs during upload.
    """
    api = HfApi()
    try:
        logging.info(f"Uploading folder '{folder_path}' to Hugging Face repo '{repo_id}' ({repo_type})...")
        api.upload_large_folder(folder_path=folder_path, repo_id=repo_id, repo_type=repo_type)
        logging.info("Upload completed successfully.")
    except Exception as e:
        logging.error(f"Upload failed: {e}")
        raise


def parse_arguments():
    """Parses command-line arguments for file or folder upload."""
    parser = argparse.ArgumentParser(description="Upload files or folders to Hugging Face Hub.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the file or folder to upload."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face repository ID."
    )
    parser.add_argument(
        "--repo_type",
        type=str,
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Type of the Hugging Face repository (default: dataset)."
    )
    return parser.parse_args()




if __name__ == "__main__":
    args = parse_arguments()
    path = args.path

    if os.path.isfile(path):
        upload_file_to_huggingface(file_path=path, repo_id=args.repo_id, repo_type=args.repo_type)
    elif os.path.isdir(path):
        upload_to_huggingface(folder_path=path, repo_id=args.repo_id, repo_type=args.repo_type)
    else:
        logging.error(f"The provided path '{path}' is neither a file nor a folder.")
