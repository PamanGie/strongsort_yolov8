# StrongSORT with YOLOv8 Tracking

## Overview
This project implements StrongSORT object tracking using YOLOv8 for object detection, integrated via BoxMOT library.

![OC-SORT Tracking Demo](https://i.postimg.cc/sgk0CM1P/Screenshot-2025-04-03-114555.png)

## Prerequisites

### System Requirements
- CUDA-capable GPU
- Python 3.8+
- pip package manager

### Dependencies
The project requires the following libraries:

| Package | Version | Compatibility |
|---------|---------|---------------|
| torch | 2.2.1+cu121 | ≥2.2.1, <3.0.0 ✅ |
| torchvision | 0.17.1+cu121 | ≥0.17.1, <0.18.0 ✅ |
| torchaudio | 2.2.1+cu121 | 2.2.1+cu121 ✅ |
| opencv-python | 4.7.0.72 | Any ✅ |
| scipy | 1.9.3 | Any ✅ |
| ultralytics | 8.3.100 | Latest ✅ |

### Installation

1. Clone the repository:
```bash
git clone https://github.com/PamanGie/strongsort_yolov8
cd strongsort-yolov8
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download pre-trained weights:
- YOLOv8 weights: `yolov8n.pt`
- OSNet weights: `osnet_x0_25_msmt17.pt`

## Usage

```python
from ultralytics import YOLO
from boxmot import StrongSORT

# Load YOLO and StrongSORT
model = YOLO('yolov8n.pt')
tracker = StrongSORT('osnet_x0_25_msmt17.pt')

# Tracking logic here
```

## Performance Notes
- Tested on RTX 3060
- Average FPS: 2-3 fps
- Not recommended for edge devices like Jetson

## Limitations
- High computational overhead
- Significant FPS drop compared to DeepSORT/ByteTrack
- Cross-version compatibility challenges

## License
This project follows the licensing terms of the [BoxMOT repository](https://github.com/mikel-brostrom/boxmot). 
Please refer to the original BoxMOT repository for detailed licensing information.

## Contribution
Contributions are welcome. Please open an issue or submit a pull request.
