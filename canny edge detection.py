import cv2
import os
import matplotlib.pyplot as plt

# Path ke folder gambar train (hasil download dari Roboflow)
image_folder = "dataset/rail_segmentation/train/images"
image_folder = "dataset/rail_segmentation/train/images"
# Cek apakah folder dataset ada
if not os.path.exists(image_folder):
    print("❌ ERROR: Folder dataset tidak ditemukan!")
    print("Pastikan kamu sudah jalankan 'python download_dataset.py'")
    exit()

# Ambil semua file gambar (.jpg, .png) di folder tersebut
images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Jika tidak ada gambar, hentikan program
if not images:
    print("❌ Tidak ada gambar di folder train/images!")
    exit()

# Ambil gambar pertama
first_image = os.path.join(image_folder, images[0])
print(f"✅ Memproses gambar: {first_image}")

# Baca gambar dalam mode grayscale (hitam putih)
img = cv2.imread(first_image, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("❌ Gagal membaca gambar — mungkin file rusak atau format tidak didukung.")
    exit()

# Proses Canny Edge Detection
blurred = cv2.GaussianBlur(img, (5, 5), 0)  # Blur untuk kurangi noise
edges = cv2.Canny(blurred, 50, 150)         # Deteksi tepi

# Simpan hasil ke file
cv2.imwrite("canny_result.jpg", edges)
print("✅ Hasil disimpan sebagai: canny_result.jpg")

# Tampilkan hasil di VS Code (gunakan matplotlib — aman, tidak error)
plt.figure(figsize=(8, 6))
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')  # Hilangkan sumbu x dan y
plt.show()       # Tampilkan gambar

print("✅ Program selesai. Gambar juga tersimpan di folder ini sebagai 'canny_result.jpg'.")