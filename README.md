# Supplier Performance & PO Prediction Dashboard

Dashboard analitik berbasis Streamlit untuk memprediksi **Lead Time** dan **kemungkinan keterlambatan Purchase Order (PO)** menggunakan machine learning.  
Data yang digunakan bersifat **sintetis**, dibangkitkan secara programatik menggunakan library [`Faker`](https://faker.readthedocs.io/), dan **dirancang mengikuti logika bisnis nyata** guna menciptakan skenario yang realistis—termasuk variasi lead time berdasarkan kategori produk, region supplier, dan tingkat keandalan supplier.

Live Demo : https://lead-time-po-late-prediction.streamlit.app/

---

## 🎯 Tujuan
- Memprediksi **lead time aktual** sebelum PO dikirim.
- Memprediksi **probabilitas keterlambatan (Is Late)** suatu PO.
- Memberikan **rekomendasi berbasis risiko** dan visualisasi **performa historis supplier**.
- Menghitung metrik operasional kunci seperti **OTIF (On-Time In-Full)**.

> 💡 **Catatan Data**:  
> Karena data riil tidak tersedia, dataset dibangkitkan dengan pola bisnis yang masuk akal:
> - Supplier di region tertentu (misal: "Asia") cenderung memiliki lead time lebih panjang.
> - Kategori produk seperti "Personal Care" lebih rentan terlambat daripada "Beverage".
> - Supplier dengan `Reliability` rendah memiliki tingkat keterlambatan (`Is_Late = True`) dan variasi lead time yang lebih tinggi.
> - Nama supplier dihasilkan secara acak menggunakan `Faker` untuk menjaga anonimitas dan realisme.

---

## 🧠 Model Machine Learning

| Komponen       | Tipe Model      | Algoritma Terbaik | Target                     |
|----------------|------------------|-------------------|----------------------------|
| Lead Time      | Regresi          | **SVR**           | `Lead_Time_Days` (hari)    |
| Keterlambatan  | Klasifikasi      | **XGBoost**       | `Is_Late` (True/False)     |

> Model dilatih menggunakan fitur yang mencerminkan dinamika bisnis nyata:
> - **Statistik historis supplier**: `Supplier_Avg_LT`, `Supplier_Late_Rate`, `Supplier_Reliability`
> - **Karakteristik PO**: `Quantity_Ordered`, `Order_Date`, `Expected_Lead_Time`
> - **Metadata supplier**: `Category` (Food, Beverage, Household, Personal Care), `Region` (Asia, Europe, Americas), `Base_Price`
> - **Encoding kategorikal**: `Supplier_ID_TE` (Target Encoding berbasis performa historis)
>
> Pola keterlambatan dan lead time **tidak acak**, melainkan dikendalikan oleh logika generasi data:
