"""
Fase 0 - Sanity Check
Bukan bagian dari VLA pipeline. Cuma buat verifikasi bahwa:
1. OpenCV ke-install dengan benar
2. Webcam laptop bisa diakses Python

Jalankan: python fase0_webcam_check.py
Tekan 'q' di window video untuk keluar.
"""

import cv2


def main():
    # Index 0 = webcam default. Kalau laptop punya lebih dari satu kamera
    # (misal ada eksternal), coba ganti ke 1, 2, dst.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Webcam nggak bisa dibuka.")
        print("Cek: (1) webcam nggak lagi dipakai app lain,")
        print("     (2) driver webcam OK,")
        print("     (3) izin kamera di Windows Settings > Privacy > Camera nyala.")
        return

    print("[OK] Webcam berhasil dibuka.")
    print(
        f"     Resolusi default: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    )
    print("     Tekan 'q' di window video untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal baca frame dari webcam.")
            break

        cv2.imshow("Fase 0 - Webcam Check (tekan 'q' untuk keluar)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] Webcam check selesai, semua resource sudah dilepas.")


if __name__ == "__main__":
    main()
