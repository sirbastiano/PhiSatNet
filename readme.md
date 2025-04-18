# 📦 🛰️ Data Format Specification (DFS)

- Designed by Roberto Del Prete and Nicolas Longépé, Apr-7-2025

## ⚠️ Warning

As a rule of thumb, **do not include data augmentation in the dataset**. Always store the original data to ensure reproducibility and flexibility for downstream tasks. Augmentation should be applied dynamically during training or preprocessing.



## 🛠️ Tools & Dependencies

This repository defines the structure and usage of a cloud optimised zarr dataset format designed for machine learning tasks such as classification, segmentation, regression, compression, and reconstruction. The format adopted here is the one of the PhiSatNet paper.

- Install requirements:
```
pip install zarr numpy
```
    
>	🧩 zarr: Chunked, compressed, N-dimensional arrays for Python. \
>	🧩 numpy: For data handling.


## 📁 Dataset Structure

The dataset is organized in two main groups:

```
<dataset_root>.zarr/
├── TrainVal/
│   ├── <sample_id>/
│   │   ├── img/            # Image array
│   │   ├── label/          # Label array or map
│   │   └── metadata/       # Metadata attributes
│   └── …
└── Test/
    └── <sample_id>/
        ├── img/
        ├── label/
        └── metadata/
```

Each `<sample_id>` corresponds to a unique tile (or chip).


If cross-validation sets are available, adopt a **hierarchical structure** to organize the dataset, ensuring clear separation between training, validation, and testing subsets.

### 📂 Hierarchical Cross-Validation Structure

For datasets requiring **cross-validation**, the following hierarchical structure is recommended:

```
<dataset_root>.zarr/
├── Fold_1/
│   ├── TrainVal/
│   │   ├── <sample_id>/
│   │   │   ├── img/
│   │   │   ├── label/
│   │   │   └── metadata/
│   └── Test/
│       ├── <sample_id>/
│       │   ├── img/
│       │   ├── label/
│       │   └── metadata/
├── Fold_2/
│   ├── TrainVal/
│   └── Test/
└── Fold_N/
  ├── TrainVal/
  └── Test/
```

Each `Fold_X` directory represents a cross-validation fold, with separate `Train` and `Val` subsets. This structure ensures clear organization and reproducibility for cross-validation experiments.

---

## 🖼️ Image Data

- **Type**: `zarr.array`
- **Shape**: `(C, H, W)`
- **Dtype**: `float32`
- **Description**: Multi-channel image data (e.g., satellite image with RGB or spectral bands).

---

## 🏷️ Label Formats

Depending on the task, labels are stored in one of the following formats:


| **Task**                          | **Shape**      | **Dtype**              | **Value Description**                            |
|----------------------------------|----------------|------------------------|--------------------------------------------------|
| 🔹 Classification                | `()`           | `int`                  | Class label ∈ `[0, N]`                           |
| 🔹 Segmentation                  | `(H, W)`       | `int`                  | Pixel-wise class ID map ∈ `[0, N_classes]`       |
| 🔹 Regression / Compression / Reconstruction | `(C, H, W)`   | `float32` or `uint8`   | Continuous or discrete value maps                |

---

## 🧭 Metadata

Metadata is stored under the `metadata/` directory as attributes or nested dictionaries. The following fields are supported:

### 🔹 Task Descriptor
- **`task`**  
  - **Type**: `string`  
  - **Values**: `"classification"`, `"segmentation"`, `"regression"`, `"compression"`  
  - **Description**: Defines the machine learning task associated with the sample.

### 🔹 Sensor Information
- **`sensor`**: Satellite/sensor identifier (e.g., `"S2A"`)  
- **`sensor_resolution`**: Ground resolution in meters (e.g., `10`)  
- **`sensor_orbit`**: Orbit direction (e.g., `"ASCENDING"`)  
- **`spectral_bands_ordered`**: Ordered spectral bands (e.g., `"B2-B3-B4-B4"`)  
- **`sensor_orbit_number`**: Orbit number identifier (e.g., `123`)

### 🔹 Acquisition Timestamp
- **`datatake`**  
  - **Format**: `"dd-mm-yyyy HH:MM:SS"`  
  - **Example**: `"07-04-2025 12:30:00"`

### 🔹 Geolocation
    
- **`crs`**  
  - **Type**: string of Coordinate Reference System  
  - **Example**:
    ```json
        'EPSG:4326'
    ```
    

- **`geolocation`**  
  - **Type**: Dictionary of corner coordinates  
  - **Example**:
    ```json
    {
        "UL": [45.0, 10.0],
        "UR": [45.0, 10.1],
        "LL": [44.9, 10.0],
        "LR": [44.9, 10.1]
    }
    ```

### 🔹 Ancillary Information
- **`cloud_cover`**: Cloud percentage (e.g., `12.3`)  
- **`sun_azimuth`**: Sun azimuth angle in degrees  
- **`sun_elevation`**: Sun elevation angle in degrees  
- **`view_azimuth`**: Sensor azimuth angle in degrees  
- **`view_elevation`**: Sensor elevation angle in degrees

---

## ✅ A) Dataset Creation Example (Python)

> Follow the **dataset_creation.ipynb** notebook example.

## 🚀 B) Usage Example: Loading and Iterating Over the Dataset

This example demonstrates how to load the Zarr-based dataset for the segmentation task using a custom PyTorch DataLoader.

```python
from data_loader import get_zarr_dataloader, NormalizeChannels
from tqdm import tqdm

# Dataset configuration
zarr_path = "burned_area_dataset.zarr"
dataset_set = "trainval"

# Initialize DataLoader
dataloader = get_zarr_dataloader(
    zarr_path=zarr_path,
    dataset_set=dataset_set,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    transform=NormalizeChannels(min_max=True),
    task_filter="segmentation",
    metadata_keys=["sensor", "timestamp"],
)

# Iterate through batches
for batch in tqdm(dataloader, desc="Processing Batches"):
    for task in batch['tasks']:
        images = batch[f'{task}_img']
        labels = batch[f'{task}_label']
        # Your training loop or inference logic here
```

## 🌐 Uploading to Hugging Face

To upload the dataset to Hugging Face, you can use the provided `zip_n_push.sh` script. This script automates the process of zipping the dataset and pushing it to the Hugging Face Hub.

### Steps to Upload:
1. Ensure the dataset is organized as described in the **Dataset Structure** section.
2. Run the `zip_n_push.sh` script:
  ```bash
  source zip_n_push.sh <dataset_root>.zarr <huggingface_repo_url>
  ```
  Replace `<dataset_root>.zarr` with the path to your dataset and `<huggingface_repo_url>` with the URL of your Hugging Face repository.

### Example:
```bash
bash zip_n_push.sh burned_area_dataset.zarr https://huggingface.co/username/burned_area_dataset
```

### Notes:
- Ensure you have the necessary permissions to push to the Hugging Face repository.
- The script requires `git-lfs` to be installed and configured for handling large files.
- For more details, refer to the Hugging Face documentation: [Hugging Face Hub](https://huggingface.co/docs/hub).





⸻

## 📝 License

> This dataset structure is released under the Creative Commons Attribution (CC-BY) license. Any dataset created or shared using this format must retain the same license, ensuring proper attribution and enabling reuse and extension for both academic and commercial purposes.