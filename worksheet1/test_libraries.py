# test_libs.py

print("=== TEST GENERAL LIBRARIES ===")
try:
    import numpy as np
    import pandas as pd
    print("NumPy version:", np.__version__)
    print("Pandas version:", pd.__version__)
except Exception as e:
    print("General libs error:", e)

print("\n=== TEST AUDIO LIBRARIES ===")
try:
    import librosa
    import soundfile as sf
    import scipy
    print("Librosa version:", librosa.__version__)
    print("SoundFile version:", sf.__version__)
    print("Scipy version:", scipy.__version__)
except Exception as e:
    print("Audio libs error:", e)

print("\n=== TEST IMAGE LIBRARIES ===")
try:
    import cv2
    from PIL import Image
    import skimage
    import matplotlib
    print("OpenCV version:", cv2.__version__)
    print("Pillow version:", Image.__version__)
    print("scikit-image version:", skimage.__version__)
    print("Matplotlib version:", matplotlib.__version__)
except Exception as e:
    print("Image libs error:", e)

print("\n=== TEST VIDEO LIBRARY ===")
try:
    import moviepy
    print("MoviePy version:", moviepy.__version__)
except Exception as e:
    print("Video lib error:", e)

print("\n✅ Semua library berhasil di-import tanpa error!")
