
# 📦 🛰️ Dataset Format Specification

- Roberto Del Prete, Apr-7-2025

This repository defines the structure and usage of a cloud optimised zarr dataset format designed for machine learning tasks such as classification, segmentation, regression, compression, and reconstruction.

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

Each `<sample_id>` corresponds to a unique tile.

---

## 🖼️ Image Data

- **Type**: `zarr.array`
- **Shape**: `(C, H, W)`
- **Dtype**: `float32`
- **Description**: Multi-channel image data (e.g., satellite image with RGB or spectral bands).

---

## 🏷️ Label Formats

Depending on the task, labels are stored in one of the following formats:

### 🔹 Classification
- **Shape**: `()` (scalar)
- **Dtype**: `int`
- **Value**: Class label ∈ `[0, N]`

### 🔹 Segmentation
- **Shape**: `(H, W)`
- **Dtype**: `int`
- **Value**: Pixel-wise class ID map ∈ `[0, N_classes]`

### 🔹 Regression / Compression / Reconstruction
- **Shape**: `(C, H, W)`
- **Dtype**: `float32` or `uint8`
- **Value**: Continuous or discrete value maps

---

## 🧭 Metadata

Stored under `metadata/` as attributes or nested dicts. Includes:

- **`task`**:
    - Format: String
    - Values: `"classification"`, `"segmentation"`, `"regression"`, `"compression"`
    - Description: Defines the machine learning task for this sample

- **Sensor Information**:
    - **`sensor`**: Satellite/sensor identifier (e.g., `"S2A"`)
    - **`sensor_resolution`**: Ground resolution in meters (e.g., `10`)
    - **`sensor_orbit`**: Orbit direction (e.g., `"ASCENDING"`)
    - **`spectral_bands_ordered`**: Band ordering (e.g., `"B2-B3-B4-B4"`)
    - **`sensor_orbit_number`**: Numeric orbit identifier (e.g., `123`)

- **`datatake`**:  
    - Format: `"dd-mm-yyyy HH:MM:SS"`  
    - Example: `"07-04-2025 12:30:00"`

- **`geolocation`**:  
    - Format: Dict of corner coordinates  
    - Example:
        ```json
        {
            "UL": [45.0, 10.0],
            "UR": [45.0, 10.1],
            "LL": [44.9, 10.0],
            "LR": [44.9, 10.1]
        }
        ```

- **Ancillary Information**:
    - **`cloud_cover`**: Percentage (e.g., `12.3`)
    - **`sun_azimuth`**: Angle in degrees
    - **`sun_elevation`**: Angle in degrees
    - **`view_azimuth`**: Angle in degrees
    - **`view_elevation`**: Angle in degrees

---

## ✅ Example (Python)

```python
import zarr
import numpy as np

root = zarr.open("my_dataset.zarr", mode="w")

sample = root.create_group("TrainVal/00001")
sample.create_dataset("img", data=np.random.rand(3, 256, 256).astype(np.float32))
sample.create_dataset("label", data=np.random.randint(0, 5))  # classification label

# Metadata
sample.attrs["datatake"] = "07-04-2025 12:30:00"
sample.attrs["geolocation"] = {
    "UL": [45.0, 10.0],
    "UR": [45.0, 10.1],
    "LL": [44.9, 10.0],
    "LR": [44.9, 10.1]
}
sample.attrs["ancillary"] = {
    "sensor": "S2A",
    "cloud_cover": 12.3
}
```


⸻

## 🛠️ Tools & Dependencies
    
    •	zarr: Chunked, compressed, N-dimensional arrays for Python.
    •	numpy: For data handling.

Install requirements:
```
pip install zarr numpy
```



⸻

## 🧩 Use Cases
    •	Supervised classification and segmentation
    •	Regression from spatial inputs
    •	Image compression and reconstruction tasks
    •	Satellite Earth Observation data pipelines

⸻

## 📝 License

This dataset structure is open for use and extension under the CC-BY license.

⸻

For questions or contributions, feel free to open an issue or pull request!