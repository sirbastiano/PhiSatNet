# 📦 🛰️ Data Format Specification (DFS)

- Designed by Roberto Del Prete and Nicolas Longépé, Apr-7-2025

This repository defines the structure and usage of a cloud optimised zarr dataset format designed for machine learning tasks such as classification, segmentation, regression, compression, and reconstruction. The format adopted here is the one of the PhiSatNet paper.

## 🛠️ Tools & Dependencies

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


⸻

## 📝 License

> This dataset structure is released under the Creative Commons Attribution (CC-BY) license. Any dataset created or shared using this format must retain the same license, ensuring proper attribution and enabling reuse and extension for both academic and commercial purposes.