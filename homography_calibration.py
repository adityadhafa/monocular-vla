"""
Fase 1.3a - Kalibrasi Homography
Layer 1 (Perception) - fondasi mapping pixel -> koordinat fisik (cm).

Cara pakai:
1. Taruh 4 penanda di MEJA (bisa kertas/tape) yang membentuk persegi
   panjang, dengan ukuran yang lo tahu persis (misal 40cm x 30cm).
2. Jalankan script ini, lalu klik 4 titik itu di gambar webcam
   SESUAI URUTAN yang diminta di terminal (kiri-atas, kanan-atas,
   kanan-bawah, kiri-bawah).
3. Masukkan ukuran nyata area kerja lo (lebar x tinggi dalam cm).
4. Matriks homography disimpan ke file 'homography.npy', dipakai lagi
   nanti oleh pixel_to_world_demo.py tanpa perlu kalibrasi ulang
   (kecuali kamera/posisi meja berubah).

Jalankan: python homography_calibration.py
"""

import cv2
import numpy as np

clicked_points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        print(f"  Titik {len(clicked_points)} dicatat: ({x}, {y})")


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam nggak bisa dibuka.")
        return

    window_name = "Fase 1.3a - Kalibrasi (klik 4 titik sesuai urutan)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    labels = ["KIRI-ATAS", "KANAN-ATAS", "KANAN-BAWAH", "KIRI-BAWAH"]
    print("Klik 4 sudut area kerja lo di window video, urutannya:")
    for i, label in enumerate(labels, 1):
        print(f"  {i}. {label}")
    print("Tekan 'r' buat reset kalau salah klik. Tekan 'q' buat batal.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Gambar titik-titik yang udah diklik + urutan koneksinya
        for i, pt in enumerate(clicked_points):
            cv2.circle(frame, pt, 6, (0, 255, 255), -1)
            cv2.putText(
                frame,
                str(i + 1),
                (pt[0] + 10, pt[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        if len(clicked_points) > 1:
            for i in range(1, len(clicked_points)):
                cv2.line(
                    frame, clicked_points[i - 1], clicked_points[i], (0, 255, 255), 1
                )

        # Instruksi live di layar
        if len(clicked_points) < 4:
            next_label = labels[len(clicked_points)]
            cv2.putText(
                frame,
                f"Klik titik: {next_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                "4 titik lengkap! Tekan ENTER buat lanjut.",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            clicked_points.clear()
            print("Direset. Klik ulang dari titik 1.")
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print("Dibatalkan.")
            return
        elif key == 13 and len(clicked_points) == 4:  # ENTER
            break

    cap.release()
    cv2.destroyAllWindows()

    # Minta ukuran fisik area kerja
    print("\nMasukkan ukuran NYATA area kerja lo (dalam cm).")
    width_cm = float(input("  Lebar (jarak KIRI-ATAS ke KANAN-ATAS) dalam cm: "))
    height_cm = float(input("  Tinggi (jarak KIRI-ATAS ke KIRI-BAWAH) dalam cm: "))

    # Titik tujuan dalam koordinat dunia nyata (cm), urutan harus konsisten
    # dengan urutan klik: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah
    src_points = np.array(clicked_points, dtype=np.float32)
    dst_points = np.array(
        [
            [0, 0],
            [width_cm, 0],
            [width_cm, height_cm],
            [0, height_cm],
        ],
        dtype=np.float32,
    )

    homography_matrix, _ = cv2.findHomography(src_points, dst_points)

    np.save("homography.npy", homography_matrix)
    print("\n[OK] Matriks homography disimpan ke 'homography.npy'")
    print(f"     Area kerja: {width_cm}cm x {height_cm}cm")
    print("     Sekarang jalankan pixel_to_world_demo.py untuk verifikasi.")


if __name__ == "__main__":
    main()
