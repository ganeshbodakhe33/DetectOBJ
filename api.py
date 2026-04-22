from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
from detector.model import ObjectDetectionModel
from detector.inference import detect_objects
from detector.utils import load_labels
from detector.logger import setup_logger

# Initialize FastAPI app
app = FastAPI(title="AI Detection API", version="2.0")

# Setup logger
logger = setup_logger()

# Load model ONCE (very important for performance)
model_loader = ObjectDetectionModel(
    "models/frozen_inference_graph.pb",
    "models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
)
model = model_loader.get_model()

# Load labels
labels = load_labels("models/labels.txt")


# ================= HEALTH CHECK =================
@app.get("/")
def health_check():
    """
    Check if API is running
    """
    return {"status": "API is running 🚀"}


# ================= IMAGE DETECTION =================
@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = Query(0.5),
    classes: str = Query(None)
):
    """
    Detect objects in image
    
    file -> uploaded image
    confidence -> threshold
    classes -> comma separated filter (e.g., "person,car")
    """

    try:
        # Read file bytes
        contents = await file.read()

        # Convert to numpy array
        np_arr = np.frombuffer(contents, np.uint8)

        # Decode image
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Parse class filter
        selected_classes = classes.split(",") if classes else None

        # Run detection
        output, results, count, fps = detect_objects(
            model, image, labels, confidence, selected_classes
        )

        # Convert output image to base64 (for frontend display)
        _, buffer = cv2.imencode(".jpg", output)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        logger.info("Image processed successfully")

        return JSONResponse(content={
            "detections": results,
            "count": count,
            "fps": fps,
            "image": img_base64
        })

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ================= VIDEO DETECTION =================
@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    """
    Detect objects in video (basic version)
    """

    try:
        # Save temp video
        with open("temp_video.mp4", "wb") as f:
            f.write(await file.read())

        cap = cv2.VideoCapture("temp_video.mp4")

        frame_results = []

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            _, results, _, _ = detect_objects(
                model, frame, labels, 0.5, None
            )

            frame_results.append(results)

        cap.release()

        return {
            "frames_processed": len(frame_results),
            "results": frame_results[:10]  # limit response size
        }

    except Exception as e:
        logger.error(f"Video Error: {str(e)}")
        return {"error": str(e)}