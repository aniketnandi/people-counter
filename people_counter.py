"""
People detection and line-crossing counting with classical computer vision.

Pipeline:
  1. Read frames from a video file or webcam and resize for speed.
  2. Detect people with OpenCV's built-in HOG + linear SVM person detector.
  3. Filter detections by confidence, size, and aspect ratio, then apply
     non-maximum suppression (NMS) to remove overlapping boxes.
  4. Track people with a centroid tracker. A track is only "confirmed"
     after it is matched in MIN_HITS consecutive frames, which removes
     flickering one-frame false positives.
  5. Count each confirmed person once when their current position is on
     the opposite side of the counting line from where they were first
     seen (robust to missed frames, unlike frame-to-frame crossing checks).

Usage:
  python people_counter.py --video path/to/clip.mp4
  python people_counter.py --video clip.mp4 --line 0.8
  python people_counter.py              # uses the webcam

Requires OpenCV 4.x (pip install "opencv-python<5"); HOGDescriptor is not
in the main OpenCV 5 package.

Keys:
  q  - quit
"""

import argparse
import time

import cv2
import numpy as np

FRAME_WIDTH = 640           # resize width; HOG is slow on large frames
SCORE_THRESHOLD = 0.5       # minimum HOG SVM confidence
NMS_THRESHOLD = 0.4
MIN_ASPECT, MAX_ASPECT = 1.5, 4.0   # people are ~2-3x taller than wide
MIN_BOX_HEIGHT = 60         # ignore tiny far-away detections (pixels)
MAX_MATCH_DISTANCE = 75     # max pixels a centroid can move between frames
MAX_MISSED_FRAMES = 15      # drop a track after this many missed frames
MIN_HITS = 5                # frames a track must persist before it is confirmed


class CentroidTracker:
    """Assigns persistent IDs to detections by nearest-centroid matching."""

    def __init__(self):
        self.next_id = 0
        self.tracks = {}

    def update(self, detections):
        """detections: list of (centroid, box). Returns the track dict."""
        unmatched = list(range(len(detections)))
        for tid, track in list(self.tracks.items()):
            if unmatched:
                dists = [np.linalg.norm(np.subtract(track["centroid"], detections[i][0]))
                         for i in unmatched]
                best = int(np.argmin(dists))
                if dists[best] <= MAX_MATCH_DISTANCE:
                    i = unmatched.pop(best)
                    track["centroid"], track["box"] = detections[i]
                    track["missed"] = 0
                    track["hits"] += 1
                    continue
            track["missed"] += 1
            if track["missed"] > MAX_MISSED_FRAMES:
                del self.tracks[tid]

        for i in unmatched:
            centroid, box = detections[i]
            self.tracks[self.next_id] = {"centroid": centroid, "box": box,
                                         "start": centroid, "missed": 0,
                                         "hits": 1, "counted": False}
            self.next_id += 1
        return self.tracks


def detect_people(hog, frame):
    """Run HOG, filter by confidence/shape/size, then apply NMS."""
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8),
                                          padding=(8, 8), scale=1.05)
    if len(boxes) == 0:
        return []
    rects, scores = [], []
    for (x, y, w, h), s in zip(boxes, np.array(weights).flatten()):
        if (s >= SCORE_THRESHOLD and h >= MIN_BOX_HEIGHT
                and MIN_ASPECT <= h / w <= MAX_ASPECT):
            rects.append([int(x), int(y), int(w), int(h)])
            scores.append(float(s))
    if not rects:
        return []
    keep = cv2.dnn.NMSBoxes(rects, scores, score_threshold=SCORE_THRESHOLD,
                            nms_threshold=NMS_THRESHOLD)
    return [rects[i] for i in np.array(keep).flatten()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="path to a video file (default: webcam)")
    parser.add_argument("--line", type=float, default=0.5,
                        help="counting line height as a fraction of the frame (0-1)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source.")

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    tracker = CentroidTracker()
    count_down, count_up = 0, 0
    prev_time, fps = time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        scale = FRAME_WIDTH / frame.shape[1]
        frame = cv2.resize(frame, (FRAME_WIDTH, int(frame.shape[0] * scale)))
        line_y = int(frame.shape[0] * args.line)

        boxes = detect_people(hog, frame)
        detections = [((x + w // 2, y + h // 2), (x, y, w, h)) for x, y, w, h in boxes]
        tracks = tracker.update(detections)

        for tid, t in tracks.items():
            if t["hits"] < MIN_HITS or t["missed"] > 0:
                continue                      # only draw/count confirmed, visible tracks
            (sx, sy), (cx, cy) = t["start"], t["centroid"]
            if not t["counted"] and (sy < line_y) != (cy < line_y):
                if cy > sy:
                    count_down += 1
                else:
                    count_up += 1
                t["counted"] = True
            x, y, w, h = t["box"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {tid}", (int(cx) - 15, int(cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 0, 0), 2)
        cv2.putText(frame, f"Down: {count_down}  Up: {count_up}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("People Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final counts -> down: {count_down}, up: {count_up}")


if __name__ == "__main__":
    main()
