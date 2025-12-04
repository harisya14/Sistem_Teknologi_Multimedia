# Real-time Remote Photoplethysmography (rPPG)  

**Nama**: Harisya Miranti  
**NIM**: 122140049  
**Mata Kuliah**: Sistem Teknologi Multimedia  
**Institut Teknologi Sumatera**

## 📌 Deskripsi
Implementasi sistem *real-time rPPG* yang memanfaatkan webcam untuk mendeteksi detak jantung (*BPM*) tanpa kontak fisik. Sistem mengekstraksi variasi reflektansi kulit dari wajah, memproses sinyal melalui pipeline rPPG, dan memperkirakan BPM secara real-time.

---

## ✅ Implementasi Program

### 1. **Pipeline Dasar rPPG (30%)**
- **Deteksi Wajah & ROI**: `MediaPipe Face Mesh` untuk pelacakan landmark presisi pada bagian pipi.
- **Ekstraksi Sinyal**: **Metode POS (Plane-Orthogonal-to-Skin)** — kombinasi kanal R, G, B dengan vektor ortogonal terhadap perubahan cahaya.
- **Pemrosesan Sinyal**:
  - Detrending via *sliding average*  
  - Bandpass filter Butterworth (0.67–4.0 Hz / 40–240 BPM)
- **Estimasi BPM**: FFT pada rentang frekuensi valid → frekuensi dominan → konversi ke BPM.

### 2. **Kemampuan Real-time (25%)**
- Input langsung dari **webcam live** (`cv2.VideoCapture` + `CAP_DSHOW`).
- *Sliding window* dengan buffer dinamis (`BUFFER_SIZE = FPS × WINDOW_SEC`).
- Estimasi BPM diperbarui **setiap ~1 detik**.
- Visualisasi dalam **satu jendela terintegrasi**:
  - Kiri: streaming kamera + ROI highlight  
  - Kanan: sinyal POS terfilter + spektrum FFT

### 3. **Peningkatan Kualitas / Improvement**
| Peningkatan | Penjelasan |
|------------|------------|
|  **Penggunaan Metode POS** | Mengganti rata-rata kanal hijau  ke **metode POS**, yang dirancang untuk ketahanan terhadap gerakan dan perubahan pencahayaan  |
|  **ROI berbasis landmark spesifik** | ROI dibentuk dari **10 landmark** (`[227, 123, ..., 230]`) yang mengcover area pipi sehigga deteksi lebih stabil daripada bounding box penuh. |
|  **Skin segmentation** | Filter HSV (`H∈[0,25]`, `S∈[30,150]`, `V∈[60,255]`) untuk mengekstrak hanya pixel kulit di dalam ROI — mengurangi noise dari rambut, alis, dan latar belakang. |
|  **Visualisasi informatif** | Plot real-time sinyal (time-domain) dan spektrum frekuensi (FFT, frequency-domain) dalam satu figure — memudahkan interpretasi dan debugging. |

### 4. **Kualitas Kode**
- Modular: konfigurasi, inisialisasi, fungsi pemrosesan, dan loop utama tersusun secara terpisah.
- Komentar jelas dan dokumentasi fungsi (khususnya `pos_algorithm`).
- Parameter mudah dikonfigurasi (`FPS`, `WINDOW_SEC`, rentang BPM).
- Kode siap jalan tanpa dependensi eksternal selain pustaka standar.

---

## 🚀 Cara Menjalankan
```bash
# Instal dependensi
pip install opencv-python mediapipe numpy scipy matplotlib

# Jalankan
python rppg_realtime.py

# tekan 'q' untuk keluar