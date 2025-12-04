import cv2
import mediapipe as mp
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from collections import deque
import time

# ───────────────────────────────────────────────
# 1. KONFIGURASI
# ───────────────────────────────────────────────
FPS = 20
WINDOW_SEC = 8
BUFFER_SIZE = FPS * WINDOW_SEC
BPM_MIN, BPM_MAX = 40, 240
FREQ_MIN, FREQ_MAX = BPM_MIN / 60, BPM_MAX / 60  # 0.67–4 Hz

# Buffers: simpan R, G, B per frame (untuk POS)
R_buffer = deque(maxlen=BUFFER_SIZE)
G_buffer = deque(maxlen=BUFFER_SIZE)
B_buffer = deque(maxlen=BUFFER_SIZE)
pos_signal_buffer = deque(maxlen=BUFFER_SIZE)

# ───────────────────────────────────────────────
# 2. INISIALISASI MEDIAPIPE
# ───────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

ROI_LANDMARKS = [227, 123, 137, 147, 187, 206, 203, 142, 120, 230]

# ───────────────────────────────────────────────
# 3. PLOTTING: 1 WINDOW, KIRI-KANAN
# ───────────────────────────────────────────────
plt.ion()
fig = plt.figure(figsize=(13, 5))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

ax_cam = fig.add_subplot(gs[0, 0])
ax_cam.set_title("Camera")
ax_cam.axis("off")
img_display = ax_cam.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

ax_signal = fig.add_subplot(gs[0, 1])
ax_signal.set_title("POS Signal + FFT")
ax_signal.set_xlabel("Time (s)")
ax_signal.set_ylabel("Amplitude")
ax_signal.grid(True)

line_signal, = ax_signal.plot([], [], 'g-', linewidth=1.2, label="POS Signal")
line_fft, = ax_signal.plot([], [], 'b--', linewidth=1.0, label="FFT")
ax_signal.legend(loc="upper right")

current_bpm = "--"
fig.suptitle("Real-time rPPG (POS Method) | Est. BPM: --", fontsize=14, fontweight="bold")

# ───────────────────────────────────────────────
# 4. FUNGSI PEMROSESAN SINYAL
# ───────────────────────────────────────────────
def bandpass_filter(data, lowcut, highcut, fs, order=3):
    ny = 0.5 * fs
    b, a = signal.butter(order, [lowcut/ny, highcut/ny], btype='band')
    return signal.filtfilt(b, a, data)

def sliding_detrend(data, window=15):
    if len(data) < window:
        return data
    trend = np.convolve(data, np.ones(window)/window, mode='same')
    return data - trend

def estimate_bpm(data, fs):
    if len(data) < 30:
        return None

    detr = sliding_detrend(data)
    filtered = bandpass_filter(detr, FREQ_MIN, FREQ_MAX, fs)

    N = len(filtered)
    yf = np.abs(fft(filtered)[:N//2])
    xf = fftfreq(N, 1/fs)[:N//2]

    mask = (xf >= FREQ_MIN) & (xf <= FREQ_MAX)
    if not np.any(mask):
        return None

    idx = np.argmax(yf[mask])
    freq_est = xf[mask][idx]
    bpm = freq_est * 60

    # Update FFT plot
    line_fft.set_xdata(xf[mask])
    line_fft.set_ydata(yf[mask])

    return bpm, filtered

#  IMPLEMENTASI METODE POS 
def pos_algorithm(R, G, B, alpha=0.67):
    """
    Plane-Orthogonal-to-Skin (POS) Method
    Input: R, G, B = list/array nilai rata-rata per frame
    Output: sinyal rPPG skalar
    """
    R, G, B = np.array(R), np.array(G), np.array(B)
    if len(R) == 0 or len(G) == 0 or len(B) == 0:
        return np.array([])

    # Normalisasi kanal (mengurangi pengaruh cahaya global)
    mean_R, mean_G, mean_B = np.mean(R), np.mean(G), np.mean(B)
    if mean_R == 0 or mean_G == 0 or mean_B == 0:
        return np.zeros_like(R)

    R_norm = R / mean_R
    G_norm = G / mean_G
    B_norm = B / mean_B

    # POS vector: [α, 1, -α] — ortogonal terhadap arah perubahan cahaya
    C = np.vstack([R_norm, G_norm, B_norm]).T  # N × 3
    H = np.array([alpha, 1, -alpha])
    S = np.dot(C, H)  # N × 1

    return S

# ───────────────────────────────────────────────
# 5. BUKA KAMERA
# ───────────────────────────────────────────────
def open_camera(idx=0):
    print(f"[INFO] Membuka kamera {idx}...", end="", flush=True)
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("❌ Kamera gagal dibuka")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    time.sleep(0.5)
    print(" OK")
    return cap

# ───────────────────────────────────────────────
# 6. MAIN LOOP — DENGAN METODE POS
# ───────────────────────────────────────────────
if __name__ == "__main__":
    try:
        try:
            cap = open_camera(1)
        except:
            print("[INFO] Mencoba kamera internal...")
            cap = open_camera(0)

        last_update = time.time()
        print("Tekan 'q' pada jendela Python untuk keluar.")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            roi_signal = None
            if results.multi_face_landmarks:
                h, w = frame.shape[:2]
                pts = []
                for idx in ROI_LANDMARKS:
                    lm = results.multi_face_landmarks[0].landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    pts.append([x, y])
                pts = np.array(pts)

                hull = cv2.convexHull(pts)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, hull, 255)

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                skin = cv2.inRange(hsv, (0, 30, 60), (25, 150, 255))
                final = cv2.bitwise_and(mask, skin)

                # ✅ Ekstraksi R, G, B rata-rata dari ROI kulit
                R_vals = frame[:, :, 2][final > 0]
                G_vals = frame[:, :, 1][final > 0]
                B_vals = frame[:, :, 0][final > 0]

                if len(R_vals) > 10:
                    R_mean = np.mean(R_vals)
                    G_mean = np.mean(G_vals)
                    B_mean = np.mean(B_vals)

                    R_buffer.append(R_mean)
                    G_buffer.append(G_mean)
                    B_buffer.append(B_mean)

                    # Hitung sinyal POS jika buffer cukup
                    if len(R_buffer) >= 3:  # POS butuh minimal 3 titik
                        pos_sig = pos_algorithm(list(R_buffer), list(G_buffer), list(B_buffer))
                        if len(pos_sig) > 0:
                            roi_signal = pos_sig[-1]  # ambil nilai terbaru
                            pos_signal_buffer.append(roi_signal)

                # Visualisasi ROI
                overlay = frame.copy()
                overlay[final > 0] = (0, 255, 0)
                frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
                cv2.polylines(frame, [hull], True, (0, 255, 0), 2)

            # Update BPM dari buffer POS (bukan green)
            if len(pos_signal_buffer) > FPS*3 and (time.time() - last_update) > 1:
                arr = np.array(pos_signal_buffer)
                result = estimate_bpm(arr, FPS)

                if result is not None:
                    bpm, filtered = result
                    current_bpm = f"{bpm:.1f}"

                    t = np.arange(len(filtered)) / FPS
                    line_signal.set_xdata(t)
                    line_signal.set_ydata(filtered)

                    ax_signal.relim()
                    ax_signal.autoscale_view()

                    fig.suptitle(f"Real-time rPPG (POS) | Est. BPM: {current_bpm}",
                                 fontsize=14, fontweight="bold")

                last_update = time.time()

            # Update tampilan kamera
            img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            fig.canvas.draw()
            fig.canvas.flush_events()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nDihentikan.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.close()
        print("Selesai.")