# Animal Classifier Web App

Web interface untuk model Animal Classifier berbasis PyTorch + Flask.

## Cara Menjalankan

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Letakkan model

Pastikan file `animal_classifier_optimized.pth` berada di folder yang sama dengan `app.py`.
File ini dihasilkan setelah training selesai di Google Colab.

### 3. Jalankan server

```bash
python app.py
```

### 4. Buka browser

Akses: `http://localhost:5000`

## Struktur Folder

```
web_app/
├── app.py                          # Flask backend
├── requirements.txt                # Python dependencies
├── animal_classifier_optimized.pth # Model weights (dari Colab)
└── website/
    └── index.html                  # Frontend UI
```

## Fitur

- Upload gambar via drag & drop atau file browser
- Prediksi real-time: Cat, Dog, Wild
- Tampilkan confidence score untuk semua kelas
- UI modern dengan dark theme
