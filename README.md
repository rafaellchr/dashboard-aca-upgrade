# 📊 Insurance Analytics Dashboard

Dashboard interaktif untuk memvisualisasikan data asuransi, memonitor sebaran premi, dan menganalisis performa transaksi secara real-time. Dibangun menggunakan Python dan Streamlit.

## 🚀 Demo Aplikasi
**[Klik di sini untuk melihat Dashboard Live](https://dashboard-aca-bogor-latest.streamlit.app/)**

## 🌟 Fitur Utama
* **High Performance Data Loading:** Menggunakan format `.parquet` untuk memproses puluhan ribu data transaksi dengan cepat (menggantikan CSV konvensional).
* **Interactive Filtering:** User dapat memfilter data berdasarkan Sidebar.
* **Distribution Analysis:** Visualisasi sebaran nilai premi menggunakan Histogram untuk mendeteksi pola transaksi nasabah.
* **KPI Tracking:** Menampilkan metrik utama (Total Premi, Jumlah Transaksi, Rata-rata) secara instan.

## 🛠️ Teknologi yang Digunakan
* **Python 3.x**
* **Streamlit** (Frontend Dashboard)
* **Pandas** (Data Manipulation)
* **Plotly/Altair** (Data Visualization)
* **Parquet** (Data Storage Optimization)

## 📂 Struktur Folder
Agar aplikasi berjalan lancar, pastikan struktur folder seperti berikut:
```text
.
├── app.py                  # Main application code
├── requirements.txt        # Python dependencies
├── README.md               # Documentation
└── data_produksi/          # Folder khusus data
    └── data_dashboard.parquet
