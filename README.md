# Supplier Performance & PO Prediction Dashboard

Dashboard analitik berbasis Streamlit untuk memprediksi **Lead Time** dan **kemungkinan keterlambatan Purchase Order (PO)** menggunakan machine learning. Dirancang untuk mendukung pengambilan keputusan logistik dan manajemen supplier secara proaktif.

---

## 🎯 Tujuan
- Memprediksi **lead time aktual** sebelum PO dikirim.
- Memprediksi **probabilitas keterlambatan (Is Late)** suatu PO.
- Memberikan **rekomendasi berbasis risiko** dan menampilkan **performa historis supplier**.
- Menghitung metrik bisnis kunci seperti **OTIF (On-Time In-Full)**.

---

## 🧠 Model Machine Learning

| Komponen       | Tipe Model      | Algoritma Terbaik | Target                     |
|----------------|------------------|-------------------|----------------------------|
| Lead Time      | Regresi          | **SVR**           | `Lead_Time_Days` (hari)    |
| Keterlambatan  | Klasifikasi      | **XGBoost**       | `Is_Late` (True/False)     |

> Model dilatih menggunakan fitur seperti:
> - Statistik historis supplier (`Avg_LT`, `Late_Rate`, `Reliability`)
> - Informasi PO (`Quantity_Ordered`, `Order_Date`, `Expected_Delivery_Date`)
> - Metadata supplier (`Category`, `Region`, `Base_Price`)
> - Fitur encoding kategorikal (`Supplier_ID_TE` via Target Encoding)