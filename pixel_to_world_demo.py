"""
Fase 1.3b - Pixel-to-World Demo
Layer 1 (Perception) - menyatukan color detection + homography.

Prasyarat: sudah jalanin homography_calibration.py dan file
'homography.npy' sudah ada di folder yang sama.

Ini membuktikan pipeline lengkap Fase 1: webcam -> deteksi warna ->
koordinat pixel -> koordinat fisik (cm) di atas meja.

Jalankan: python pixel_to_world_demo.py
Tekan 'q' untuk keluar.
"""

import cv2
import numpy as np
import os

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


def detect_largest_blob(hsv_frame, ranges):
    mask = None
    for lower, upper in ranges:
        m = cv2.inRange(hsv_frame, lower, upper)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


def pixel_to_world(homography_matrix, px, py):
    """
    Transform satu titik pixel (px, py) ke koordinat dunia nyata (cm)
    menggunakan matriks homography hasil kalibrasi.
    """
    point = np.array([[[px, py]]], dtype=np.float32)
    world_point = cv2.perspectiveTransform(point, homography_matrix)
    wx, wy = world_point[0][0]
    return wx, wy


def main():
    if not os.path.exists("homography.npy"):
        print("[ERROR] File 'homography.npy' nggak ketemu.")
        print("        Jalankan homography_calibration.py dulu.")
        return

    homography_matrix = np.load("homography.npy")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam nggak bisa dibuka.")
        return

    print("[OK] Pixel-to-world demo jalan. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for color_name, ranges in COLOR_RANGES.items():
            result = detect_largest_blob(hsv, ranges)
            if result is None:
                continue
            px, py = result
            wx, wy = pixel_to_world(homography_matrix, px, py)

            color = DISPLAY_COLOR[color_name]
            cv2.circle(frame, (px, py), 6, color, -1)
            cv2.putText(
                frame,
                f"{color_name}: ({wx:.1f}cm, {wy:.1f}cm)",
                (px + 10, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        cv2.imshow("Fase 1.3b - Pixel to World (tekan 'q' untuk keluar)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
