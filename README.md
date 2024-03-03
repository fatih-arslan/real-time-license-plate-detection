# Real-time License Plate Recognition System with YOLOv8 and Flask

This project implements a real-time license plate recognition system using YOLOv8 for object detection and Flask for creating a web-based interface. The system is designed to detect license plates in real-time video streams and display the results on a user-friendly web application.

## Features

- **Object Detection**: The project utilizes the YOLOv8 algorithm for real-time object detection.
- **License Plate Recognition**: After detecting license plates in the video stream, the application uses Optical Character Recognition (OCR) to extract text from the license plate images.
- **Web Interface**: The application provides a user-friendly web interface where users can upload a video file or access their webcam for real-time object detection.
- **Real-time Processing**: Provides real-time license plate recognition capabilities for live video streams.

## Requirements

- Python 3.x
- Flask
- Ultralytics YOLO
- OpenCV
- Pytesseract

## Installation

1. Clone the repository:

```bash
git clone https://github.com/fatih-arslan/real-time-license-plate-detection.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the YOLOv8 weights file (e.g., `best.pt`) and place it in the project directory.

4. Install Tesseract OCR and specify its path in the `YOLO_Video.py` file:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## Usage

1. Run the Flask application:

```bash
python flask-app.py
```

2. Upload a video file or grant permission to access the webcam for object detection.

3. The application will detect license plates in the video stream and display the results in real-time.
