from ultralytics import YOLO
from boxmot import StrongSORT
from pathlib import Path
import torch
import cv2
import numpy as np

# Cek ketersediaan CUDA
print(torch.cuda.is_available())

# Load model YOLO dengan GPU
model = YOLO('yolov8n.pt').to('cuda')

# Dapatkan nama kelas
class_names = model.names

# Cetak daftar kelas untuk referensi
print("Daftar Kelas YOLO:")
for idx, class_name in enumerate(class_names):
    print(f"{idx}: {class_name}")

# Gunakan Path untuk model weights
model_weights = Path('osnet_x0_25_msmt17.pt')

# Inisialisasi StrongSORT dengan GPU
tracker = StrongSORT(
    model_weights=model_weights,
    device='0',
    fp16=True
)

# Buka video
cap = cv2.VideoCapture('video.mp4')

# Dictionary untuk menyimpan track class
track_classes = {}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Deteksi dengan YOLO
    results = model(frame)
    
    # Extract detections
    bboxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    
    # Modifikasi format deteksi untuk StrongSORT
    dets = np.column_stack([
        bboxes,    # koordinat
        scores,    # confidence
        classes    # class
    ])
    
    # Update tracker
    tracks = tracker.update(dets, frame)
    
    # Visualisasi tracking
    for track in tracks:
        bbox = track[:4]
        track_id = int(track[4])
        
        # Cari kelas asli untuk track ini dari deteksi YOLO
        # Cari deteksi terdekat dengan track
        best_match_idx = np.argmin(np.sum((bboxes[:, :4] - bbox)**2, axis=1))
        class_id = int(classes[best_match_idx])
        class_name = class_names[class_id]
        
        # Simpan kelas untuk track ID ini
        track_classes[track_id] = class_name
        
        # Pilih warna berdasarkan kelas
        if class_name == 'car':
            color = (0, 0, 255)  # Merah untuk mobil
        elif class_name == 'person':
            color = (0, 255, 0)  # Hijau untuk orang
        elif class_name == 'bus':
            color = (255, 0, 0)  # Biru untuk bus
        elif class_name == 'truck':
            color = (0, 255, 255)  # Kuning untuk truck
        else:
            color = (128, 128, 128)  # Abu-abu untuk kelas lainnya
        
        # Gambar kotak dan ID dengan nama kelas
        cv2.rectangle(
            frame, 
            (int(bbox[0]), int(bbox[1])), 
            (int(bbox[2]), int(bbox[3])), 
            color, 
            2
        )
        cv2.putText(
            frame, 
            f'ID: {track_id} {class_name}', 
            (int(bbox[0]), int(bbox[1]) - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            color, 
            2
        )
    
    # Tampilkan frame
    cv2.imshow('Tracking', frame)
    
    # Keluar dengan menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup video dan jendela
cap.release()
cv2.destroyAllWindows()
