import cv2
import time

def detect_objects(model, image, class_labels, conf_threshold=0.5, selected_classes=None):
    start_time = time.time()  # Start timer

    classIndex, confidence, bbox = model.detect(image, confThreshold=conf_threshold)

    results = []
    detection_count = {}

    if len(classIndex) != 0:
        for classInd, conf, box in zip(classIndex.flatten(), confidence.flatten(), bbox):

            label = class_labels[classInd - 1]

            # 🎯 FILTER FEATURE
            if selected_classes and label not in selected_classes:
                continue

            # Count objects
            detection_count[label] = detection_count.get(label, 0) + 1

            results.append({
                "label": label,
                "confidence": float(conf),
                "box": box.tolist()
            })

            # Draw box
            cv2.rectangle(image, box, (255, 0, 0), 2)

            cv2.putText(
                image,
                f"{label} {round(conf,2)}",
                (box[0] + 10, box[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # ⚡ FPS calculation
    fps = 1 / (time.time() - start_time)

    return image, results, detection_count, fps