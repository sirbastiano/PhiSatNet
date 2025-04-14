#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: source zip_n_push.sh <dataset_root>.zarr <huggingface_repo_url>"
    exit 1
fi

# Parse input arguments
DATASET_NAME="$1"
REPO_URL="$2"

# Extract repository ID and type from the URL
REPO_ID=$(echo "$REPO_URL" | awk -F '/' '{print $(NF-1)"/"$NF}')
REPO_TYPE="dataset"

# Define constants
DATASET_DIR=$(dirname "$DATASET_NAME")
ZIP_NAME="$(basename "$DATASET_NAME").zip"
MAX_ZIP_SIZE="10g"
UPLOAD_SCRIPT="up.py"

# Navigate to the dataset directory
cd "$DATASET_DIR" || { echo "Error: Failed to navigate to $DATASET_DIR"; exit 1; }

# Verify the dataset directory exists
if [ ! -d "$(basename "$DATASET_NAME")" ]; then
    echo "Error: Directory '$(basename "$DATASET_NAME")' does not exist."
    exit 1
fi

# Compress the dataset into split zip files
7z a -tzip -v"$MAX_ZIP_SIZE" "$ZIP_NAME" "$(basename "$DATASET_NAME")/."
# or use the zipper.py script
# pip install py7zr
# python zipper.py --input_dir "$(basename "$DATASET_NAME")" --output_dir "$DATASET_DIR" --max_size "$MAX_ZIP_SIZE"


# Upload the compressed dataset to Hugging Face
python "$UPLOAD_SCRIPT" --path "./$ZIP_NAME" --repo_id "$REPO_ID" --repo_type "$REPO_TYPE"
