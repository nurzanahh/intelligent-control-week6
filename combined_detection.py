import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Pastikan file model tersedia di lokasi yang benar
MODEL_PATH = "best.pt"

# Memuat model dengan YOLO jika berasal dari Ultralytics
try:
    model = YOLO(MODEL_PATH)
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

# Fungsi utama
def detect_canny_edges_with_model(low_threshold=50, high_threshold=150):
    cap = cv2.VideoCapture(0)  # Gunakan kamera default
    
    if not cap.isOpened():
        print("Error: Kamera tidak dapat diakses!")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Gagal membaca frame dari kamera!")
            break
        
        # Konversi ke grayscale untuk Canny
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Konversi frame ke RGB untuk YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Prediksi menggunakan model YOLO
        results = model(frame_rgb)
        predictions = results[0]

        # Buat mask kosong untuk menandai area objek model
        mask_model = np.zeros_like(edges)

        # Buat overlay transparan untuk segmentasi warna
        overlay = frame.copy()

        # Warna yang akan digunakan:
        SEGMENTATION_COLOR = (255, 0, 0) # Biru (BGR) untuk Segmentasi
        LABEL_BOX_COLOR = (255, 255, 255) # Putih (BGR) untuk teks dan frame

        # --- Bagian Kritis: Visualisasi Segmentasi dan Bounding Box ---
        
        # Cek apakah model menggunakan segmentasi (mask)
        if predictions.masks is not None and predictions.masks.xy:  # Model segmentasi
            for i, mask_polygon in enumerate(predictions.masks.xy):
                mask = np.array(mask_polygon, np.int32)
                
                # 1. Mengisi area objek dengan warna biru transparan (Area Segmentasi)
                cv2.fillPoly(overlay, [mask], SEGMENTATION_COLOR)
                
                # Tambahkan ke mask untuk mengubah tepi Canny
                cv2.fillPoly(mask_model, [mask], 255)

                # Dapatkan koordinat Bounding Box dari mask untuk label
                x1, y1, x2, y2 = map(int, predictions.boxes.xyxy[i])

                # 2. Tambahkan Label dan Frame Teks (Sesuai gaya gambar Anda)
                if hasattr(predictions.boxes, "cls") and hasattr(predictions.boxes, "conf"):
                    conf = predictions.boxes.conf[i].item()
                    label_idx = int(predictions.boxes.cls[i].item())
                    # Asumsikan ID kelas 'rail' adalah 0 (atau ganti dengan nama model.names)
                    label_name = model.names.get(label_idx, "rail-detection")
                    label = f"{label_name} {conf:.2f}"
                    
                    # Hitung ukuran kotak teks
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Frame Label (biru pekat, sama dengan segmentasi)
                    cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), SEGMENTATION_COLOR, -1)
                    
                    # Teks Label (putih)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LABEL_BOX_COLOR, 2)
                    
                    # Gambar Bounding Box (biru tipis di frame)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), SEGMENTATION_COLOR, 2)

        elif predictions.boxes is not None and len(predictions.boxes.xyxy) > 0:  # Model hanya deteksi objek
            for i, box in enumerate(predictions.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Tandai area objek pada mask
                cv2.rectangle(mask_model, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)

                # 1. Mengisi area bounding box dengan warna biru transparan
                cv2.rectangle(overlay, (x1, y1), (x2, y2), SEGMENTATION_COLOR, thickness=cv2.FILLED)

                # 2. Tambahkan Label dan Frame Teks
                if hasattr(predictions.boxes, "cls") and hasattr(predictions.boxes, "conf"):  
                    conf = predictions.boxes.conf[i].item()
                    label_idx = int(predictions.boxes.cls[i].item())
                    label_name = model.names.get(label_idx, "rail-detection")
                    label = f"{label_name} {conf:.2f}"

                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Frame Label (biru pekat)
                    cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), SEGMENTATION_COLOR, -1)
                    
                    # Teks Label (putih)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LABEL_BOX_COLOR, 2)
                    
                    # Gambar Bounding Box (biru tipis di frame)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), SEGMENTATION_COLOR, 2)


        # Gabungkan overlay dengan transparansi (disesuaikan agar segmen biru tampak jelas)
        # Transparansi 0.5 cukup dekat dengan gambar Anda
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # --- Visualisasi Canny Edges (tetap sama) ---

        # Buat hasil Canny dalam format BGR
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Ubah warna tepi Canny hanya pada area objek menjadi merah
        edges_bgr[np.where((mask_model == 255) & (edges == 255))] = [0, 0, 255]  # Tepi model merah
        edges_bgr[np.where((mask_model == 0) & (edges == 255))] = [255, 255, 255]  # Tepi lainnya tetap putih

        # Gabungkan hasil
        combined = np.hstack((frame, edges_bgr))

        # Menampilkan hasil kombinasi
        cv2.imshow('Original & Canny Edge Detection + Model Prediction (Rail-Detection)', combined)
        
        # Tekan 'q' untuk keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Jalankan program
detect_canny_edges_with_model()