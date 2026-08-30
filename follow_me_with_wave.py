"""
Find a person with the Tello camera and react to wave gestures.

Behavior:
  1. Enter SDK mode and start the video stream.
  2. Take off.
  3. Rotate in small steps until a face is visible.
  4. Watch for a side-to-side wave near the face.
  5. Alternate between moving closer and moving back on each wave.

Keep one hand on Ctrl+C while testing. The script tries to land on exit, but
real drones still need clear space, good light, and a safe test area.
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TELLO_LIB_DIR = os.path.join(PROJECT_DIR, "Tello-master")
if TELLO_LIB_DIR not in sys.path:
    sys.path.insert(0, TELLO_LIB_DIR)

from tello import (  # noqa: E402
    clockwise,
    forward,
    get_battery,
    get_video_frame,
    land,
    start,
    start_video,
    stop_video,
    takeoff,
    backward,
)

MIN_BATTERY_PERCENT = 10
ROTATE_STEP_DEGREES = 30
MAX_SEARCH_ROTATION_DEGREES = 360
MOVE_DISTANCE_CM = 40
FACE_SEARCH_SECONDS_PER_STEP = 1.5
WAVE_COOLDOWN_SECONDS = 3.0
WAVE_HISTORY_SECONDS = 1.6
WAVE_MIN_DIRECTION_CHANGES = 3
WAVE_MIN_MOTION_AREA = 650


Face = Tuple[int, int, int, int]


@dataclass
class MotionSample:
    timestamp: float
    center_x: float


class WaveDetector:
    def __init__(self) -> None:
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=80,
            varThreshold=28,
            detectShadows=False,
        )
        self.samples: list[MotionSample] = []
        self.last_wave_time = 0.0

    def update(self, frame, face: Face) -> bool:
        now = time.monotonic()
        x, y, w, h = face
        frame_h, frame_w = frame.shape[:2]

        roi_x1 = max(0, x - w)
        roi_y1 = max(0, y - h // 2)
        roi_x2 = min(frame_w, x + 2 * w)
        roi_y2 = min(frame_h, y + 2 * h)
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            return False

        fg_mask = self.background_subtractor.apply(roi)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        moving_regions = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > WAVE_MIN_MOTION_AREA]
        if not moving_regions:
            self._drop_old_samples(now)
            return False

        largest = max(moving_regions, key=lambda rect: rect[2] * rect[3])
        center_x = roi_x1 + largest[0] + largest[2] / 2
        self.samples.append(MotionSample(now, center_x))
        self._drop_old_samples(now)

        if now - self.last_wave_time < WAVE_COOLDOWN_SECONDS:
            return False

        if self._direction_changes() >= WAVE_MIN_DIRECTION_CHANGES:
            self.samples.clear()
            self.last_wave_time = now
            return True

        return False

    def _drop_old_samples(self, now: float) -> None:
        self.samples = [
            sample
            for sample in self.samples
            if now - sample.timestamp <= WAVE_HISTORY_SECONDS
        ]

    def _direction_changes(self) -> int:
        if len(self.samples) < 5:
            return 0

        changes = 0
        previous_direction = 0
        previous_x = self.samples[0].center_x
        for sample in self.samples[1:]:
            delta = sample.center_x - previous_x
            previous_x = sample.center_x
            if abs(delta) < 8:
                continue

            direction = 1 if delta > 0 else -1
            if previous_direction and direction != previous_direction:
                changes += 1
            previous_direction = direction

        return changes


def load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Could not load OpenCV face cascade: {cascade_path}")
    return detector


def find_largest_face(frame, detector: cv2.CascadeClassifier) -> Optional[Face]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(70, 70),
    )
    if len(faces) == 0:
        return None
    return tuple(max(faces, key=lambda face: face[2] * face[3]))


def wait_for_frame(timeout_seconds: float = 6.0):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        frame = get_video_frame()
        if frame is not None:
            return frame
        time.sleep(0.05)
    raise RuntimeError("Timed out waiting for the Tello video stream.")


def find_person_by_rotating(detector: cv2.CascadeClassifier) -> Face:
    rotated = 0
    while rotated <= MAX_SEARCH_ROTATION_DEGREES:
        deadline = time.monotonic() + FACE_SEARCH_SECONDS_PER_STEP
        while time.monotonic() < deadline:
            frame = wait_for_frame(timeout_seconds=1.0)
            face = find_largest_face(frame, detector)
            if face is not None:
                return face
            time.sleep(0.05)

        clockwise(ROTATE_STEP_DEGREES)
        rotated += ROTATE_STEP_DEGREES

    raise RuntimeError("Could not find a face after a full rotation.")


def main() -> None:
    detector = load_face_detector()
    wave_detector = WaveDetector()
    next_move_is_closer = True
    airborne = False

    try:
        print("Connecting to Tello...")
        start()

        battery = get_battery()
        print(f"Battery: {battery}%")
        if battery < MIN_BATTERY_PERCENT:
            raise RuntimeError(
                f"Battery is too low for autonomous flight. Charge above {MIN_BATTERY_PERCENT}%."
            )

        print("Starting video stream...")
        start_video()
        wait_for_frame()

        print("Taking off...")
        takeoff()
        airborne = True

        print("Searching for a face...")
        face = find_person_by_rotating(detector)
        print(f"Face found at {face}. Wave to command movement.")

        while True:
            frame = wait_for_frame(timeout_seconds=1.0)
            current_face = find_largest_face(frame, detector)
            if current_face is not None:
                face = current_face

            if wave_detector.update(frame, face):
                if next_move_is_closer:
                    print(f"Wave detected: moving closer {MOVE_DISTANCE_CM} cm")
                    print(f"Battery: {battery}%")
                    forward(MOVE_DISTANCE_CM)
                else:
                    print(f"Wave detected: moving back {MOVE_DISTANCE_CM} cm")
                    backward(MOVE_DISTANCE_CM)
                next_move_is_closer = not next_move_is_closer

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if airborne:
            print("Landing...")
            land()
        stop_video()


if __name__ == "__main__":
    main()
