# Player Re‑Identification & Tracking

This repository provides a simple pipeline to detect and re‑identify (track) players in a sports video using:

- **YOLO** (Ultralytics) for player detection  
- **ViT** (Timm) for appearance feature extraction  
- **Kalman Filter** + Hungarian assignment + fused cost (IoU, appearance similarity, center distance) for robust tracking  

---

## Contents

- `player_identify.py` — main tracking script  
- `requirements.txt` — Python dependencies  
- Example video/model paths configured inside `player_identify.py`  

---

## Quick Start

1. Clone this repository  
2. Install Python 3.13 and create a virtual environment  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # Linux / macOS
   .\.venv\Scripts\activate       # Windows PowerShell
Adjust the model_path (`MODEL_PATH`) , Input video path (`VIDEO_PATH`) and output video path (`OUTPUT_PATH`) inside `player_identify.py`
