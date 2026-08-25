"""
Fase 1.1 - Color Detector
Layer 1 (Perception) dari arsitektur VLA.

Deteksi objek berwarna merah, hijau, biru dari feed webcam,
lalu keluarkan koordinat pixel (centroid) tiap objek yang terdeteksi.

Jalankan: python color_detector.py
Tekan 'q' untuk keluar.
"""

import cv2
import numpy as np

# Rentang warna dalam HSV (Hue, Saturation, Value).
# Kenapa HSV, bukan RGB/BGR? Karena Hue (warna) terpisah dari Value
# (kecerahan) -> deteksi warna jadi jauh lebih tahan terhadap perubahan
# pencahayaan dibanding kalau kita threshold langsung di RGB.
#
# Catatan penting: Merah itu spesial di HSV karena Hue-nya "melingkar"
# (wrap around di 0/180), jadi merah butuh DUA rentang: dekat 0 dan
# dekat 180. Kalau nanti warna objek fisik lo ternyata meleset dari
# rentang default ini, itu NORMAL -> nanti kita bikin tool kalibrasi
# pakai trackbar biar bisa disesuaikan langsung, bukan tebak-tebakan angka.
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

# Warna kotak/label yang digambar di layar untuk tiap kelas (dalam BGR,
# ini cuma buat tampilan, bukan buat deteksi)
DISPLAY_COLOR = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
}

# Objek yang lebih kecil dari ini (dalam pixel^2) diabaikan -> ini
# nyaring noise kecil (pantulan cahaya, bintik warna acak, dll)
MIN_CONTOUR_AREA = 500


def detect_color(frame, hsv_frame, color_name, ranges):
    """
    Bikin mask biner untuk satu warna, cari contour terbesar,
    dan kembalikan (pixel_x, pixel_y, area) kalau ketemu objek valid.
    Return None kalau tidak ada objek yang cukup besar.
    """
    mask = None
    for lower, upper in ranges:
        m = cv2.inRange(hsv_frame, lower, upper)
        mask = m if mask is None else cv2.bitwise_or(mask, m)

    # Morphological ops buat bersihin noise kecil di mask
    # (erode = hapus bintik kecil, dilate = balikin ukuran objek asli)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < MIN_CONTOUR_AREA:
        return None

    # Centroid pakai image moments (cara standar OpenCV buat titik tengah)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return cx, cy, area, largest


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam nggak bisa dibuka.")
        return

    print("[OK] Color detector jalan. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections = {}

        for color_name, ranges in COLOR_RANGES.items():
            result = detect_color(frame, hsv, color_name, ranges)
            if result is None:
                continue
            cx, cy, area, contour = result
            detections[color_name] = (cx, cy)

            # Gambar bounding box, centroid, dan label di frame asli
            x, y, w, h = cv2.boundingRect(contour)
            color = DISPLAY_COLOR[color_name]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(
                frame,
                f"{color_name} ({cx},{cy}) area={int(area)}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Print koordinat ke terminal tiap frame (ini yang nanti jadi
        # "output" Layer 1 buat dikonsumsi Layer 2 / planner)
        if detections:
            print(detections)

        cv2.imshow("Fase 1.1 - Color Detector (tekan 'q' untuk keluar)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
