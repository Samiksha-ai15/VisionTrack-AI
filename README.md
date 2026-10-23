# 🔍 VisionTrack AI — Object Detection and Tracking

VisionTrack AI is a computer vision project designed to detect and track objects in visual input using the YOLO object detection model. It provides an interactive interface for exploring object detection and tracking results.

## ✨ Features

* 🎯 **Object Detection:** Detect objects using a YOLO model.
* 📹 **Object Tracking:** Track detected objects across video frames, where supported by the application.
* 🖥️ **Interactive Interface:** Use a Streamlit-based interface to interact with the project.
* 📊 **Visual Results:** View detection results through the application.
* ⚡ **Deep Learning:** Uses the Ultralytics YOLO framework.

## 🛠️ Technologies Used

* Python
* YOLO (Ultralytics)
* OpenCV
* Streamlit
* Pillow
* Pandas
* Plotly

## 📁 Project Structure

```text
VisionTrack-AI/
├── app.py
├── requirements.txt
├── yolo11n.pt
└── src/
    ├── __init__.py
    ├── detector.py
    └── test_detector.py
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Samiksha-ai15/VisionTrack-AI.git
cd VisionTrack-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use the application.

## 🎯 Project Objective

The objective of VisionTrack AI is to explore real-time computer vision, object detection, and object tracking using deep learning and Python-based tools.

## 🚀 Future Improvements

* Improve detection accuracy and performance.
* Support additional image and video input options.
* Enhance tracking and result visualization.
* Add performance metrics and detection summaries.

## 👩‍💻 Author

**Samiksha-ai15**

GitHub: https://github.com/Samiksha-ai15

## 📄 License

No license has been specified yet.
