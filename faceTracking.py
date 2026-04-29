import cv2
import mediapipe as mp
import numpy as np
from enum import Enum
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ====================== CONFIG ======================
CAMERA_INDEX = 0
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480

DETECTOR_UPSCALE = 1.5
CROP_UPSCALE = 3.5

CENTER_THRESHOLD = 0.2
# ===================================================

# === FACE DETECTOR ===
detector_base_options = python.BaseOptions(
    model_asset_path=r".\landmarks\blaze_face_full_range.tflite"
)
detector_options = vision.FaceDetectorOptions(
    base_options=detector_base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_detection_confidence=0.3
)
face_detector = vision.FaceDetector.create_from_options(detector_options)

# === FACE LANDMARKER ===
landmarker_base_options = python.BaseOptions(
    model_asset_path=r".\landmarks\face_landmarker.task"
)
landmarker_options = vision.FaceLandmarkerOptions(
    base_options=landmarker_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.2
)
face_landmarker = vision.FaceLandmarker.create_from_options(landmarker_options)

# Landmark indices
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]
LEFT_EYE_LIDS = [159, 145]
RIGHT_EYE_LIDS = [386, 374]

KEY_POINTS = LEFT_IRIS + RIGHT_IRIS + LEFT_EYE_CORNERS + RIGHT_EYE_CORNERS

class SimpleLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class trackingState(Enum):
    SEARCH_FACE = 0,
    FOUND_FACE = 1,
    FOLLOW_FACE = 2,

def get_face_size(face_landmarks_list, w, h):
    if len(face_landmarks_list) <= 454:
        return 0
    left = face_landmarks_list[234]
    right = face_landmarks_list[454]
    return abs(int(right.x * w) - int(left.x * w))

def draw_landmarks(frame, face_landmarks, w, h):
    for idx in KEY_POINTS:
        if idx < len(face_landmarks):
            x = int(face_landmarks[idx].x * w)
            y = int(face_landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

def process_eye(face, iris_idxs, corner_idxs, lid_idxs, frame, w, h):
    iris_pts = np.array([
        (int(face[i].x * w), int(face[i].y * h))
        for i in iris_idxs if i < len(face)
    ], dtype=np.int32)

    if len(iris_pts) < 3:
        return "Unknown"

    (cx, cy), _ = cv2.minEnclosingCircle(iris_pts)
    cx, cy = int(cx), int(cy)
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    eye_points = [
        (int(face[i].x * w), int(face[i].y * h))
        for i in (corner_idxs + lid_idxs) if i < len(face)
    ]

    eye_cx = int(np.mean([p[0] for p in eye_points]))
    eye_cy = int(np.mean([p[1] for p in eye_points]))

    cv2.circle(frame, (eye_cx, eye_cy), 4, (255, 0, 0), -1)

    # Normalize movement
    eye_width = max(1, abs(eye_points[1][0] - eye_points[0][0]))
    dx = (cx - eye_cx) / eye_width
    dy = (cy - eye_cy) / eye_width

    if abs(dx) < CENTER_THRESHOLD:
        horiz = "Center"
    elif dx > 0:
        horiz = "Right"
    else:
        horiz = "Left"

    if abs(dy) < CENTER_THRESHOLD:
        vert = "Center"
    elif dy > 0:
        vert = "Down"
    else:
        vert = "Up"

    return f"{horiz}-{vert}"

print("Starting Two-Stage Eye Tracker (Far Distance Optimized)")
print("Press 'q' to quit")

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

def eyeTracking():
    while True:
        startTime = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        originalHeight, originalWidth = frame.shape[:2]

        # Upscale for detection
        detectionFrame = cv2.resize(frame, None, fx=DETECTOR_UPSCALE, fy=DETECTOR_UPSCALE)
        rgbDetection = cv2.cvtColor(detectionFrame, cv2.COLOR_BGR2RGB)
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbDetection)

        timestampMs = int(time.time() * 1000)
        detectorResult = face_detector.detect_for_video(mpImage, timestampMs)

        bestFace = None
        trackedSize = 0
        gazeText = "No face"

        if detectorResult.detections:
            bestArea = 0
            bestBbox = None

            for detection in detectorResult.detections:
                bbox = detection.bounding_box

                x = int(bbox.origin_x / DETECTOR_UPSCALE)
                y = int(bbox.origin_y / DETECTOR_UPSCALE)
                boxWidth = int(bbox.width / DETECTOR_UPSCALE)
                boxHeight = int(bbox.height / DETECTOR_UPSCALE)

                x = max(0, x)
                y = max(0, y)
                boxWidth = max(1, boxWidth)
                boxHeight = max(1, boxHeight)

                area = boxWidth * boxHeight
                if area > bestArea:
                    bestArea = area
                    bestBbox = (x, y, boxWidth, boxHeight)

            if bestBbox:
                x, y, boxWidth, boxHeight = bestBbox

                cropX1 = max(0, x)
                cropY1 = max(0, y)
                cropX2 = min(originalWidth, x + boxWidth)
                cropY2 = min(originalHeight, y + boxHeight)

                face_crop = frame[cropY1:cropY2, cropX1:cropX2]

                if face_crop.size > 0:
                    crop_h, crop_w = face_crop.shape[:2]
                    targetSize = int(max(crop_w, crop_h) * CROP_UPSCALE)

                    face_crop_resized = cv2.resize(
                        face_crop, (targetSize, targetSize)
                    )

                    rgb_crop = cv2.cvtColor(face_crop_resized, cv2.COLOR_BGR2RGB)
                    mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)

                    landmarker_result = face_landmarker.detect_for_video(
                        mp_crop, timestampMs
                    )

                    if landmarker_result.face_landmarks:
                        raw_landmarks = landmarker_result.face_landmarks[0]

                        bestFace = []

                        for landmark in raw_landmarks:
                            px = landmark.x * targetSize
                            py = landmark.y * targetSize

                            origX = (px / targetSize) * (cropX2 - cropX1) + cropX1
                            origY = (py / targetSize) * (cropY2 - cropY1) + cropY1

                            normX = origX / originalWidth
                            normY = origY / originalHeight

                            bestFace.append(SimpleLandmark(normX, normY))

                        trackedSize = get_face_size(bestFace, originalWidth, originalHeight)

        if bestFace:
            left = process_eye(bestFace, LEFT_IRIS, LEFT_EYE_CORNERS, LEFT_EYE_LIDS, frame, originalWidth, originalHeight)
            right = process_eye(bestFace, RIGHT_IRIS, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS, frame, originalWidth, originalHeight)

            draw_landmarks(frame, bestFace, originalWidth, originalHeight)

            gazeText = left if left == right else f"L:{left} | R:{right}"

        fps = 1 / (time.time() - startTime)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, f"Gaze: {gazeText}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if trackedSize > 0:
            cv2.putText(frame, f"Face size: {trackedSize}px", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Eye Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
        eyeTracking()

if __name__ == "__main__":
    main()