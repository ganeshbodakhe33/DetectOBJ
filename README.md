# 🎯 DetectOBJ — AI Object Detection System

DetectOBJ is a real-time AI-powered object detection system built using OpenCV DNN and Streamlit with a scalable architecture.

It supports:
- Image Detection
- Webcam Live Detection
- Video Processing
- Class Filtering
- Performance Monitoring (FPS)

--------------------------------------------------

## 🚀 Features

### Core Features
- Upload Image & Detect Objects
- Real-time Webcam Detection
- Video File Processing
- Adjustable Confidence Threshold
- Filter Specific Object Classes

### Advanced Features
- Detection Count Dashboard
- FPS (Frames Per Second) Monitoring
- JSON Export of Results
- Download Processed Image
- Model Caching (Fast performance)
- Error Handling (Production safe)

--------------------------------------------------

## 📁 Project Structure

DetectGPT/
│
├── app.py                 # Streamlit UI
├── api.py                 # FastAPI backend (optional)
│
├── detector/
│   ├── __init__.py
│   ├── model.py           # Model loading
│   ├── inference.py       # Detection logic
│   ├── utils.py           # Helper functions
│   └── logger.py          # Logging system
│
├── models/
│   ├── frozen_inference_graph.pb
│   ├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
│   └── labels.txt
│
├── outputs/
│   ├── images/
│   └── results.json
│
└── requirements.txt

--------------------------------------------------

## ⚙️ Installation

git clone <your-repo-url>
cd DetectGPT
pip install -r requirements.txt

--------------------------------------------------

## ▶️ Run the Application

### Run Streamlit UI
streamlit run app.py

### Run API (Optional)
uvicorn api:app --reload

--------------------------------------------------

## 🧪 How to Use

1. Open the app in your browser  
2. Select mode:
   - Image
   - Webcam
   - Video  
3. Upload file or start camera  
4. Adjust confidence threshold  
5. (Optional) Filter object classes  
6. View results  

--------------------------------------------------

## 📊 Output

- Bounding boxes on detected objects  
- Label + confidence score  
- Detection count  
- FPS performance  

--------------------------------------------------

## 🧠 Tech Stack

- Python
- OpenCV (DNN Module)
- Streamlit
- FastAPI
- NumPy
 
--------------------------------------------------

## 🚀 Future Improvements

- YOLOv8 integration
- Object tracking (DeepSORT)
- Cloud deployment (AWS/GCP)
- React frontend dashboard
- Authentication system

--------------------------------------------------

## 👨‍💻 Author

Ganesh

--------------------------------------------------

## ⭐ Note

This project is a strong foundation for building real-world AI products like:

- Smart Surveillance Systems
- Retail Analytics Tools
- Traffic Monitoring AI