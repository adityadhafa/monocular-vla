"""
Fase 1.2 - Object Tracker
Layer 1 (Perception) lanjutan dari color_detector.py.

Bedanya dari color detector biasa: setiap objek yang terdeteksi dikasih
ID yang PERSISTEN antar frame (bukan cuma deteksi ulang dari nol tiap
frame), dan digambar trail (jejak posisi) selama objek bergerak.

Ini penting karena: kalau nanti robot bergerak dan sempat nutupin
objek lain (occlusion) selama 1-2 frame, kita nggak mau sistem
langsung nganggep objek itu "hilang" dan bikin ID baru pas dia
kelihatan lagi.

Jalankan: python object_tracker.py
Tekan 'q' untuk keluar.
"""

import cv2
import numpy as np
from collections import deque, OrderedDict

# Sama seperti color_detector.py
COLOR_RANGES = {
    "Red": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255])),
    ],
    "Green": [
        (np.array([40, 70, 70]), np.array([80, 255, 255])),
    ],
    "Blue": [
        (np.array([90, 70, 70]), np.array([130, 255, 255])),
    ],
}

DISPLAY_COLOR = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
}

MIN_CONTOUR_AREA = 500
MAX_OBJECTS_PER_COLOR = 3   # jaga-jaga kalau ada beberapa objek warna sama
TRAIL_LENGTH = 30           # berapa titik terakhir yang digambar sebagai trail
MAX_MISSED_FRAMES = 30      # toleransi berapa frame objek "hilang" sebelum ID-nya dihapus
MATCH_DISTANCE_THRESHOLD = 80  # px, jarak maksimum buat anggap centroid = objek yang sama


class CentroidTracker:
    """
    Tracker sederhana berbasis jarak Euclidean antar centroid.
    Konsepnya: tiap frame, cocokkan centroid baru dengan centroid ID
    yang sudah ada (yang jaraknya paling deket). Kalau nggak ada yang
    cocok dalam radius tertentu, itu dianggap objek baru -> ID baru.
    """

    def __init__(self):
        self.next_id = 0
        # object_id -> {"centroid": (x,y), "missed": int, "trail": deque, "color_name": str}
        self.objects = OrderedDict()

    def update(self, detections):
        """
        detections: list of (color_name, (cx, cy))
        Return: dict object_id -> info (termasuk trail) untuk digambar
        """
        # Tandai semua objek existing sebagai "belum ke-match" frame ini
        unmatched_existing = set(self.objects.keys())

        for color_name, centroid in detections:
            best_id = None
            best_dist = MATCH_DISTANCE_THRESHOLD

            # Cari objek existing dengan warna sama & jarak terdekat
            for obj_id in unmatched_existing:
                obj = self.objects[obj_id]
                if obj["color_name"] != color_name:
                    continue
                dist = np.hypot(
                    obj["centroid"][0] - centroid[0],
                    obj["centroid"][1] - centroid[1],
                )
                if dist < best_dist:
                    best_dist = dist
                    best_id = obj_id

            if best_id is not None:
                # Match ketemu -> update posisi objek yang sudah ada
                obj = self.objects[best_id]
                obj["centroid"] = centroid
                obj["missed"] = 0
                obj["trail"].append(centroid)
                unmatched_existing.discard(best_id)
            else:
                # Nggak ada match -> objek baru, kasih ID baru
                trail = deque(maxlen=TRAIL_LENGTH)
                trail.append(centroid)
                self.objects[self.next_id] = {
                    "centroid": centroid,
                    "missed": 0,
                    "trail": trail,
                    "color_name": color_name,
                }
                self.next_id += 1

        # Objek yang nggak ke-match frame ini -> tambah counter "missed"
        for obj_id in unmatched_existing:
            self.objects[obj_id]["missed"] += 1

        # Hapus objek yang udah kelamaan hilang
        to_delete = [
            obj_id for obj_id, obj in self.objects.items()
            if obj["missed"] > MAX_MISSED_FRAMES
        ]
        for obj_id in to_delete:
            del self.objects[obj_id]

        return self.objects


def find_color_blobs(hsv_frame, color_name, ranges):
    """
    Return list of (cx, cy, area) untuk SEMUA blob warna ini yang
    cukup besar (beda dari color_detector.py yang cuma ambil 1 terbesar).
    """
    mask = None
    for lower, upper in ranges:
        m = cv2.inRange(hsv_frame, lower, upper)
        mask = m if mask is None else cv2.bitwise_or(mask, m)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_CONTOUR_AREA:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        blobs.append((cx, cy, area))

    # Kalau ada lebih dari MAX_OBJECTS_PER_COLOR, ambil yang paling gede aja
    blobs.sort(key=lambda b: b[2], reverse=True)
    return blobs[:MAX_OBJECTS_PER_COLOR]


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam nggak bisa dibuka.")
        return

    tracker = CentroidTracker()
    print("[OK] Object tracker jalan. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections = []  # list of (color_name, (cx, cy))

        for color_name, ranges in COLOR_RANGES.items():
            for cx, cy, _area in find_color_blobs(hsv, color_name, ranges):
                detections.append((color_name, (cx, cy)))

        tracked_objects = tracker.update(detections)

        for obj_id, obj in tracked_objects.items():
            if obj["missed"] > 0:
                continue  # jangan gambar objek yang lagi "hilang" sementara

            color = DISPLAY_COLOR[obj["color_name"]]
            cx, cy = obj["centroid"]

            # Gambar trail (garis penghubung titik-titik posisi sebelumnya)
            trail_points = list(obj["trail"])
            for i in range(1, len(trail_points)):
                cv2.line(frame, trail_points[i - 1], trail_points[i], color, 2)

            cv2.circle(frame, (cx, cy), 6, color, -1)
            cv2.putText(
                frame,
                f"ID {obj_id} {obj['color_name']}",
                (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.imshow("Fase 1.2 - Object Tracker (tekan 'q' untuk keluar)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()