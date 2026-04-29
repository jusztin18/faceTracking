import cv2
import mediapipe as mp
import numpy as np
from enum import Enum
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ====================== CONSTANTS ======================
CAMERA_INDEX = 0
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480

DETECTOR_UPSCALE = 1.5
CROP_UPSCALE = 3.5

CENTER_THRESHOLD = 0.2
# ======================================================


detectorBaseOptions = python.BaseOptions(
    model_asset_path="blaze_face_full_range.tflite"
)
detectorOptions = vision.FaceDetectorOptions(
    base_options=detectorBaseOptions,
    running_mode=vision.RunningMode.VIDEO,
    min_detection_confidence=0.3
)
faceDetector = vision.FaceDetector.create_from_options(detectorOptions)

landmarkerBaseOptions = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)
landmarkerOptions = vision.FaceLandmarkerOptions(
    base_options=landmarkerBaseOptions,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.2
)
faceLandmarker = vision.FaceLandmarker.create_from_options(landmarkerOptions)


LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]
LEFT_EYE_LIDS = [159, 145]
RIGHT_EYE_LIDS = [386, 374]

KEY_POINTS = LEFT_IRIS + RIGHT_IRIS + LEFT_EYE_CORNERS + RIGHT_EYE_CORNERS


class SimpleLandmark:
    """! @brief Simple normalized landmark container."""
    def __init__(self, x, y):
        self.x = x
        self.y = y


class TrackingState(Enum):
    """! @brief State machine states for face tracking."""
    SEARCH_FACE = 0
    FOUND_FACE = 1
    FOLLOW_FACE = 2


def getFaceSize(faceLandmarksList, width, height):
    """!
    @brief Computes approximate face width in pixels.

    @param faceLandmarksList List of landmarks
    @param width Frame width
    @param height Frame height
    @return Face width in pixels
    """
    if len(faceLandmarksList) <= 454:
        return 0

    left = faceLandmarksList[234]
    right = faceLandmarksList[454]

    return abs(int(right.x * width) - int(left.x * width))


def drawLandmarks(frame, faceLandmarks, width, height):
    """!
    @brief Draws selected facial landmarks.

    @param frame Image frame
    @param faceLandmarks Landmark list
    @param width Frame width
    @param height Frame height
    """
    for idx in KEY_POINTS:
        if idx >= len(faceLandmarks):
            continue

        x = int(faceLandmarks[idx].x * width)
        y = int(faceLandmarks[idx].y * height)

        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)


def processEye(face, irisIdxs, cornerIdxs, lidIdxs, frame, width, height):
    """!
    @brief Estimates gaze direction for one eye.

    @param face Landmark list
    @param irisIdxs Iris indices
    @param cornerIdxs Eye corner indices
    @param lidIdxs Eyelid indices
    @param frame Image frame
    @param width Frame width
    @param height Frame height
    @return Gaze direction string
    """
    irisPoints = np.array([
        (int(face[i].x * width), int(face[i].y * height))
        for i in irisIdxs if i < len(face)
    ], dtype=np.int32)

    if len(irisPoints) < 3:
        return "Unknown"

    (cx, cy), _ = cv2.minEnclosingCircle(irisPoints)
    cx, cy = int(cx), int(cy)
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    eyePoints = [
        (int(face[i].x * width), int(face[i].y * height))
        for i in (cornerIdxs + lidIdxs) if i < len(face)
    ]

    eyeCx = int(np.mean([p[0] for p in eyePoints]))
    eyeCy = int(np.mean([p[1] for p in eyePoints]))

    cv2.circle(frame, (eyeCx, eyeCy), 4, (255, 0, 0), -1)

    eyeWidth = max(1, abs(eyePoints[1][0] - eyePoints[0][0]))
    dx = (cx - eyeCx) / eyeWidth
    dy = (cy - eyeCy) / eyeWidth

    horiz = "Center" if abs(dx) < CENTER_THRESHOLD else ("Right" if dx > 0 else "Left")
    vert = "Center" if abs(dy) < CENTER_THRESHOLD else ("Down" if dy > 0 else "Up")

    return f"{horiz}-{vert}"


def detectFaces(frame):
    """!
    @brief Runs face detection.

    @param frame Input frame
    @return Tuple(detectorResult, timestampMs)
    """
    detectionFrame = cv2.resize(frame, None, fx=DETECTOR_UPSCALE, fy=DETECTOR_UPSCALE)
    rgbDetection = cv2.cvtColor(detectionFrame, cv2.COLOR_BGR2RGB)
    mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbDetection)

    timestampMs = int(time.time() * 1000)
    detectorResult = faceDetector.detect_for_video(mpImage, timestampMs)

    return detectorResult, timestampMs


def getBestBbox(detections):
    """!
    @brief Selects largest detected face.

    @param detections Detection list
    @return Bounding box tuple or None
    """
    if not detections:
        return None

    bestArea = 0
    bestBbox = None

    for detection in detections:
        bbox = detection.bounding_box

        x = int(bbox.origin_x / DETECTOR_UPSCALE)
        y = int(bbox.origin_y / DETECTOR_UPSCALE)
        w = int(bbox.width / DETECTOR_UPSCALE)
        h = int(bbox.height / DETECTOR_UPSCALE)

        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)

        area = w * h

        if area <= bestArea:
            continue

        bestArea = area
        bestBbox = (x, y, w, h)

    return bestBbox


def stateSearch(detectorResult):
    """! @brief SEARCH state logic."""
    if not detectorResult.detections:
        return TrackingState.SEARCH_FACE, None, 0

    return TrackingState.FOUND_FACE, None, 0


def stateFound(detectorResult):
    """! @brief FOUND state logic."""
    bestBbox = getBestBbox(detectorResult.detections)

    if not bestBbox:
        return TrackingState.SEARCH_FACE, None, 0

    return TrackingState.FOLLOW_FACE, bestBbox, 0


def stateFollow(frame, detectorResult, timestampMs, width, height):
    """! @brief FOLLOW state logic."""
    bestBbox = getBestBbox(detectorResult.detections)

    if not bestBbox:
        return TrackingState.SEARCH_FACE, None, 0

    x, y, w, h = bestBbox

    cropX1 = max(0, x)
    cropY1 = max(0, y)
    cropX2 = min(width, x + w)
    cropY2 = min(height, y + h)

    faceCrop = frame[cropY1:cropY2, cropX1:cropX2]

    if faceCrop.size == 0:
        return TrackingState.SEARCH_FACE, None, 0

    cropH, cropW = faceCrop.shape[:2]
    targetSize = int(max(cropW, cropH) * CROP_UPSCALE)

    faceCropResized = cv2.resize(faceCrop, (targetSize, targetSize))
    rgbCrop = cv2.cvtColor(faceCropResized, cv2.COLOR_BGR2RGB)
    mpCrop = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbCrop)

    landmarkerResult = faceLandmarker.detect_for_video(mpCrop, timestampMs)

    if not landmarkerResult.face_landmarks:
        return TrackingState.FOUND_FACE, None, 0

    rawLandmarks = landmarkerResult.face_landmarks[0]

    bestFace = []
    for lm in rawLandmarks:
        px = lm.x * targetSize
        py = lm.y * targetSize

        origX = (px / targetSize) * (cropX2 - cropX1) + cropX1
        origY = (py / targetSize) * (cropY2 - cropY1) + cropY1

        normX = origX / width
        normY = origY / height

        bestFace.append(SimpleLandmark(normX, normY))

    trackedSize = getFaceSize(bestFace, width, height)

    return TrackingState.FOLLOW_FACE, bestFace, trackedSize


cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)


def eyeTracking():
    """!
    @brief Main tracking loop.
    """
    state = TrackingState.SEARCH_FACE

    while True:
        startTime = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        gazeText = "No face"
        bestFace = None
        trackedSize = 0

        detectorResult, timestampMs = detectFaces(frame)

        match state:
            case TrackingState.SEARCH_FACE:
                state, _, _ = stateSearch(detectorResult)

            case TrackingState.FOUND_FACE:
                state, _, _ = stateFound(detectorResult)

            case TrackingState.FOLLOW_FACE:
                state, bestFace, trackedSize = stateFollow(
                    frame, detectorResult, timestampMs, width, height
                )

        if bestFace:
            left = processEye(bestFace, LEFT_IRIS, LEFT_EYE_CORNERS, LEFT_EYE_LIDS,
                              frame, width, height)
            right = processEye(bestFace, RIGHT_IRIS, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS,
                               frame, width, height)

            drawLandmarks(frame, bestFace, width, height)
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
    """! @brief Program entry point."""
    eyeTracking()


if __name__ == "__main__":
    main()