# 🚗 Car Direction Detection - Wrong Way Alert System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

An AI-powered Computer Vision system that detects if a vehicle is moving in the **WRONG DIRECTION** in real-time. Built for smart traffic management and road safety.

### 🎯 Problem It Solves
In many roads, wrong-side driving causes 70% of accidents. This system automatically detects and alerts when a car goes in the wrong direction.

### ⚙️ How It Works
1.  **Detection:** YOLOv8 detects all cars in the frame
2.  **Tracking:** SORT algorithm assigns a unique ID to each car
3.  **Logic:** System calculates movement vector (dx). If `dx < 0` -> WRONG WAY!

### 🛠️ Tech Stack
- Python
- Ultralytics YOLOv8
- OpenCV
- SORT Tracker (Simple Online and Realtime Tracking)

### 🚀 How to Run
```bash
# Clone the repo
git clone https://github.com/nabab0516-bit/car---direction.git
cd car---direction

# Install dependencies
pip install ultralytics opencv-python numpy

# Run
python main.py
