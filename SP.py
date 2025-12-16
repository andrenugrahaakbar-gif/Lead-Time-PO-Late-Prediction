import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================
# KONFIGURASI
# =============================================
BASE = Path(".")
DATA_DIR = BASE / "Data"
MODELS_DIR = BASE / "Models"
FEATURES_DIR = BASE / "features"

# =============================================
# MUAT DATA & MODEL
# =============================================
@st.cache_data
def load_all_data():
    """Muat semua data yang diperlukan"""
    try:
        # Load supplier master
        supplier_master = pd.read_csv(DATA_DIR / "supplier_master.csv")
        
        # Load PO data
        po_df = pd.read_csv(DATA_DIR / "PO.csv", parse_dates=["Order_Date", "Expected_Delivery_Date"])
        
        # Load GR data
        gr_df = pd.read_csv(DATA_DIR / "GR.csv", parse_dates=["Actual_Delivery_Date"])
        
        return supplier_master, po_df, gr_df
    except Exception as e:
        st.error(f"❌ Gagal memuat data: {e}")
        return None, None, None

@st.cache_resource
def load_ml_models():
    """Muat model ML yang telah dilatih"""
    try:
        # Model Lead Time
        lt_model = joblib.load(MODELS_DIR / "leadtime_model.pkl")
        lt_encoder = joblib.load(MODELS_DIR / "lt_supplier_te_encoder.pkl")
        lt_features = joblib.load(FEATURES_DIR / "selected_features_LT.pkl")
        
        # Model Is Late
        is_late_model = joblib.load(MODELS_DIR / "IsLate_model.pkl")
        with open(FEATURES_DIR / "selected_features_IsLate.json") as f:
            is_late_features = json.load(f)
        
        return {
            "leadtime": {"model": lt_model, "encoder": lt_encoder, "features": lt_features},
            "is_late": {"model": is_late_model, "features": is_late_features}
        }
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat model: {e}")
        return None

@st.cache_data
def load_processed_data():
    """Muat dan proses data untuk analisis"""
    supplier_master, po_df, gr_df = load_all_data()
    
    if po_df is None or gr_df is None:
        return None, None, None
    
    # Merge data
    merged_df = pd.merge(po_df, gr_df, on="PO_ID", how="inner")
    
    # Hitung metrik performa
    merged_df["Delay_Days"] = (merged_df["Actual_Delivery_Date"] - merged_df["Expected_Delivery_Date"]).dt.days
    merged_df["Delay_Days"] = merged_df["Delay_Days"].clip(lower=0)
    merged_df["Lead_Time_Days"] = (merged_df["Actual_Delivery_Date"] - merged_df["Order_Date"]).dt.days
    merged_df = merged_df[merged_df["Lead_Time_Days"] >= 0]
    merged_df["Defect_Rate"] = np.where(
        merged_df["Quantity_Received"] > 0,
        merged_df["Defect_Qty"] / merged_df["Quantity_Received"],
        0.0
    )
    merged_df["Is_Late"] = (merged_df["Delay_Days"] > 0).astype(int)
    
    # Gabung dengan supplier master
    if supplier_master is not None:
        merged_df = merged_df.merge(supplier_master, on="Supplier_ID", how="left")
    
    return supplier_master, merged_df, po_df

@st.cache_data
def calculate_supplier_stats(merged_df):
    """Hitung statistik supplier"""
    if merged_df is None:
        return {}
    
    stats = merged_df.groupby("Supplier_ID").agg(
        Supplier_Avg_LT=("Lead_Time_Days", "mean"),
        Supplier_Late_Rate=("Is_Late", "mean"),
        Supplier_Late_Severity=("Delay_Days", "mean"),
        Supplier_Defect_Rate=("Defect_Rate", "mean"),
        Supplier_Reliability=("Is_Late", lambda x: 1 - x.mean()),
        Total_Orders=("PO_ID", "count"),
        Total_Quantity=("Quantity_Ordered", "sum")
    ).round(4).to_dict("index")
    
    return stats

def create_supplier_master_dict(supplier_master):
    """Buat dictionary supplier master yang benar"""
    if supplier_master is None:
        return {}
    
    supplier_dict = {}
    for _, row in supplier_master.iterrows():
        supplier_dict[row["Supplier_ID"]] = {
            "Base_Price": row["Base_Price"],
            "Category": row["Category"],
            "Region": row["Region"]
        }
    return supplier_dict

def prepare_leadtime_features(input_data, supplier_master_dict, supplier_stats):
    """Persiapkan fitur untuk model Lead Time"""
    df = pd.DataFrame([input_data])
    supp_id = input_data["Supplier_ID"]
    
    # Ambil info supplier
    if supp_id in supplier_master_dict:
        info = supplier_master_dict[supp_id]
        df["Base_Price"] = float(info["Base_Price"])
        df["Category"] = str(info["Category"])
        df["Region"] = str(info["Region"])
    else:
        df["Base_Price"], df["Category"], df["Region"] = 50.0, "Other", "Other"
    
    df["Expected_Lead_Time"] = (df["Expected_Delivery_Date"] - df["Order_Date"]).dt.days
    
    # Fitur tanggal lainnya
    order_date = df["Order_Date"].dt
    df["Order_Year"] = order_date.year
    df["Order_Month"] = order_date.month
    df["Order_Quarter"] = order_date.quarter
    
    # Statistik supplier - PASTIKAN SEMUA FITUR ADA
    if supp_id in supplier_stats:
        s = supplier_stats[supp_id]
        df["Supplier_Avg_LT"] = s["Supplier_Avg_LT"]
        df["Supplier_Late_Rate"] = s["Supplier_Late_Rate"]
        df["Supplier_Late_Severity"] = s.get("Supplier_Late_Severity", 0.0)
        df["Supplier_Defect_Rate"] = s.get("Supplier_Defect_Rate", 0.0)
        df["Supplier_Reliability"] = s.get("Supplier_Reliability", 1.0)
    else:
        # Default values untuk supplier baru
        df["Supplier_Avg_LT"] = 10.0
        df["Supplier_Late_Rate"] = 0.1
        df["Supplier_Late_Severity"] = 0.0
        df["Supplier_Defect_Rate"] = 0.0
        df["Supplier_Reliability"] = 1.0
    
    return df

def prepare_is_late_features(input_data, supplier_master_dict, supplier_stats):
    """Persiapkan fitur untuk model Is Late"""
    df = pd.DataFrame([input_data])
    supp_id = input_data["Supplier_ID"]
    
    # Ambil info supplier
    if supp_id in supplier_master_dict:
        info = supplier_master_dict[supp_id]
        df["Base_Price"] = float(info["Base_Price"])
        df["Category"] = str(info["Category"])
        df["Region"] = str(info["Region"])
    else:
        df["Base_Price"], df["Category"], df["Region"] = 50.0, "Other", "Other"
    
    # Fitur turunan
    df["Expected_Lead_Time"] = (df["Expected_Delivery_Date"] - df["Order_Date"]).dt.days
    df["Order_Month"] = df["Order_Date"].dt.month
    
    # Statistik supplier
    if supp_id in supplier_stats:
        s = supplier_stats[supp_id]
        df["Supplier_Avg_LT"] = s["Supplier_Avg_LT"]
        df["Supplier_Late_Rate"] = s["Supplier_Late_Rate"]
        df["Supplier_Late_Severity"] = s.get("Supplier_Late_Severity", 0.0)
        df["Supplier_Defect_Rate"] = s.get("Supplier_Defect_Rate", 0.0)
        df["Supplier_Reliability"] = s.get("Supplier_Reliability", 1.0)
    else:
        df["Supplier_Avg_LT"] = 10.0
        df["Supplier_Late_Rate"] = 0.1
        df["Supplier_Late_Severity"] = 0.0
        df["Supplier_Defect_Rate"] = 0.0
        df["Supplier_Reliability"] = 1.0
    
    return df

def predict_lead_time(input_data, models, supplier_master_dict, supplier_stats):
    """Prediksi lead time menggunakan model ML"""
    try:
        # Engineer fitur dengan expected_lt jika diberikan
        df_features = prepare_leadtime_features(input_data, supplier_master_dict, supplier_stats)
        
        # DEBUG: Cek apakah Supplier_ID ada sebelum encoding
        if "Supplier_ID" not in df_features.columns:
            st.error("❌ Supplier_ID tidak ditemukan dalam fitur")
            return None
            
        # Tambahkan Supplier_ID_TE untuk model Lead Time
        lt_encoder = models["leadtime"]["encoder"]
        df_features["Supplier_ID_TE"] = lt_encoder.transform(df_features[["Supplier_ID"]]).iloc[:, 0]
        
        # Drop Supplier_ID asli setelah encoding
        df_features = df_features.drop(columns=["Supplier_ID"])
        
        # Pastikan semua fitur yang dibutuhkan ada
        lt_features = models["leadtime"]["features"]
        missing_features = [f for f in lt_features if f not in df_features.columns]
        
        if missing_features:
            st.warning(f"⚠️ Fitur yang hilang: {missing_features}")
            # Tambahkan default values untuk fitur yang hilang
            for feature in missing_features:
                if "Supplier" in feature:
                    df_features[feature] = 0.0
                else:
                    df_features[feature] = 0
        
        # Prediksi
        X = df_features[lt_features]
        lt_model = models["leadtime"]["model"]
        predicted_lt = lt_model.predict(X)[0]
        
        return max(0, predicted_lt)  # Pastikan lead time tidak negatif
    except Exception as e:
        st.error(f"❌ Error dalam prediksi lead time: {e}")
        import traceback
        st.caption(f"🔍 Traceback: {traceback.format_exc()}")
        return None

def predict_is_late(input_data, models, supplier_master_dict, supplier_stats):
    """Prediksi keterlambatan menggunakan model ML"""
    try:
        # Engineer fitur untuk Is Late
        df_features = prepare_is_late_features(input_data, supplier_master_dict, supplier_stats)
        
        # Prediksi Is Late
        is_late_features = models["is_late"]["features"]
        X = df_features[is_late_features]
        is_late_model = models["is_late"]["model"]
        prob_late = is_late_model.predict_proba(X)[0][1]
        
        return prob_late
    except Exception as e:
        st.error(f"❌ Error dalam prediksi keterlambatan: {e}")
        return None

# =============================================
# HALAMAN 1: DASHBOARD UTAMA
# =============================================
def page_dashboard():
    st.title("🏠 Dashboard Utama - Supplier Performance Analytics")
    
    supplier_master, merged_df, po_df = load_processed_data()
    
    # Pastikan kolom Quantity_Received dan Quantity_Ordered ada
    if "Quantity_Received" in merged_df.columns and "Quantity_Ordered" in merged_df.columns:
        merged_df["In_Full"] = merged_df["Quantity_Received"] == merged_df["Quantity_Ordered"]
        merged_df["OTIF"] = (~merged_df["Is_Late"]) & merged_df["In_Full"]
        otif_rate = merged_df["OTIF"].mean() * 100
    else:
        # Fallback jika kolom tidak ada (hanya On-Time)
        otif_rate = (1 - merged_df["Is_Late"].mean()) * 100
        st.warning("⚠️ Kolom Quantity_Received tidak ditemukan. OTIF dihitung sebagai On-Time Rate saja.")

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_lt = merged_df["Lead_Time_Days"].mean()
        st.metric("Rata-rata Lead Time", f"{avg_lt:.1f} hari")

    with col2:
        late_rate = merged_df["Is_Late"].mean() * 100
        st.metric("Tingkat Keterlambatan", f"{late_rate:.1f}%")

    with col3:
        st.metric("OTIF", f"{otif_rate:.1f}%")

    with col4:
        total_suppliers = merged_df["Supplier_ID"].nunique()
        st.metric("Total Supplier", f"{total_suppliers}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Trend Chart
        monthly_lt = merged_df.groupby(merged_df["Order_Date"].dt.to_period("M")).agg({
            "Lead_Time_Days": "mean",
            "Is_Late": "mean"
        }).reset_index()
        monthly_lt["Order_Date"] = monthly_lt["Order_Date"].dt.to_timestamp()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=monthly_lt["Order_Date"], y=monthly_lt["Lead_Time_Days"], 
                      name="Lead Time", line=dict(color="blue")),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=monthly_lt["Order_Date"], y=monthly_lt["Is_Late"]*100, 
                      name="Late Rate (%)", line=dict(color="red")),
            secondary_y=True,
        )
        
        fig.update_layout(
            title="Trend Lead Time vs Tingkat Keterlambatan",
            xaxis_title="Bulan",
            hovermode="x unified"
        )
        
        fig.update_yaxes(title_text="Lead Time (hari)", secondary_y=False)
        fig.update_yaxes(title_text="Late Rate (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top 5 Supplier Terlambat")
        late_suppliers = merged_df.groupby("Supplier_ID")["Is_Late"].mean().sort_values(ascending=False).head(5)
        for supp, rate in late_suppliers.items():
            st.write(f"**{supp}**: {rate:.1%}")
    
    # Supplier Performance
    st.subheader("📊 Analisis Performa Supplier")
    supplier_perf = merged_df.groupby("Supplier_ID").agg({
        "Lead_Time_Days": "mean",
        "Is_Late": "mean",
        "PO_ID": "count"
    }).round(3).reset_index()
    
    fig = px.scatter(supplier_perf, x="Lead_Time_Days", y="Is_Late", 
                     size="PO_ID", color="Is_Late",
                     hover_data=["Supplier_ID"],
                     title="Performa Supplier: Lead Time vs Tingkat Keterlambatan",
                     labels={"Lead_Time_Days": "Rata-rata Lead Time (hari)", 
                            "Is_Late": "Tingkat Keterlambatan"})
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
                """
                <hr style='margin-top: 50px;'>
                <p style='text-align: center; color: gray;'>
                    © 2025 Andre Nugraha. All rights reserved.
                </p>
                """,
                unsafe_allow_html=True
            )

# =============================================
# HALAMAN 2: ANALISIS LEAD TIME
# =============================================
def page_lead_time_analysis():
    st.title("⏱️ Analisis Lead Time")
    
    supplier_master, merged_df, po_df = load_processed_data()
    
    if merged_df is None:
        st.error("Tidak dapat memuat data. Pastikan file data tersedia.")
        return
    
    # Statistik Lead Time
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_lt = merged_df["Lead_Time_Days"].mean()
        st.metric("Rata-rata LT", f"{avg_lt:.1f} hari")
    
    with col2:
        std_lt = merged_df["Lead_Time_Days"].std()
        st.metric("Std Dev LT", f"{std_lt:.1f} hari")
    
    with col3:
        min_lt = merged_df["Lead_Time_Days"].min()
        st.metric("LT Tercepat", f"{min_lt} hari")
    
    with col4:
        max_lt = merged_df["Lead_Time_Days"].max()
        st.metric("LT Terlama", f"{max_lt} hari")
    
    st.markdown("---")
    
    # Visualisasi Lead Time
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi Lead Time")
        fig = px.histogram(merged_df, x="Lead_Time_Days", 
                          title="Distribusi Lead Time",
                          nbins=20)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if "Category" in merged_df.columns:
            st.subheader("Lead Time per Kategori")
            fig = px.box(merged_df, x="Category", y="Lead_Time_Days",
                        title="Lead Time berdasarkan Kategori")
            st.plotly_chart(fig, use_container_width=True)
    
    # Lead Time by Supplier
    st.subheader("Lead Time per Supplier")
    supplier_lt = merged_df.groupby("Supplier_ID").agg({
        "Lead_Time_Days": ["mean", "std", "count"]
    }).round(2)
    supplier_lt.columns = ["Avg_LT", "Std_LT", "Order_Count"]
    supplier_lt = supplier_lt.sort_values("Avg_LT", ascending=False)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(supplier_lt.head(10), use_container_width=True)
    
    with col2:
        top_suppliers = supplier_lt.head(10).reset_index()
        fig = px.bar(top_suppliers, x="Supplier_ID", y="Avg_LT",
                    title="Top 10 Supplier dengan Lead Time Terlama",
                    color="Avg_LT")
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediksi Lead Time
    st.subheader("🔮 Prediksi Lead Time")

    models = load_ml_models()
    supplier_stats = calculate_supplier_stats(merged_df) if merged_df is not None else {}
    supplier_master_dict = create_supplier_master_dict(supplier_master)

    if models is None or merged_df is None:
        st.warning("⚠️ Model tidak tersedia untuk prediksi")
    else:
        with st.form("lt_prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                supplier_id = st.selectbox("Supplier ID", options=sorted(merged_df["Supplier_ID"].unique()))
            
            with col2:
                order_date = st.date_input("Tanggal PO", datetime.date.today())
                # ✅ GANTI: dari number_input jadi date_input
                expected_delivery_date = st.date_input(
                    "Tanggal Pengiriman Diharapkan", 
                    value=order_date + datetime.timedelta(days=14)
                )
            
            with col3:
                quantity = st.number_input("Quantity", min_value=1, value=100)
            
            predict_btn = st.form_submit_button("Prediksi Lead Time")
            
            if predict_btn:
                input_data = {
                    "Supplier_ID": supplier_id,
                    "Order_Date": pd.to_datetime(order_date),
                    "Expected_Delivery_Date": pd.to_datetime(expected_delivery_date),
                    "Quantity_Ordered": quantity
                }
                
                predicted_lt = predict_lead_time(input_data, models, supplier_master_dict, supplier_stats)
                
                if predicted_lt is not None:
                    expected_lt = (expected_delivery_date - order_date).days
                    diff = predicted_lt - expected_lt
                    
                    st.success(f"**Prediksi Lead Time: {predicted_lt:.1f} hari**")
                    
                    if diff > 0:
                        st.info(f"📦 Prediksi **{diff:.1f} hari lebih lama** dari lead time yang dijanjikan ({expected_lt} hari).")
                    elif diff < 0:
                        st.info(f"🚀 Prediksi **{abs(diff):.1f} hari lebih cepat** dari lead time yang dijanjikan ({expected_lt} hari).")
                    else:
                        st.info(f"✅ Prediksi **sama persis** dengan lead time yang dijanjikan ({expected_lt} hari).")
                    
                    # Historis
                    if supplier_id in supplier_stats:
                        hist_avg = supplier_stats[supplier_id]["Supplier_Avg_LT"]
                        st.write(f"📈 **Rata-rata historis supplier {supplier_id}: {hist_avg:.1f} hari**")

# =============================================
# HALAMAN 3: ANALISIS SUPPLIER & PREDIKSI PO
# =============================================
def page_supplier_analysis():
    st.title("🏭 Analisis Supplier & Prediksi PO")
    
    supplier_master, merged_df, po_df = load_processed_data()
    
    if merged_df is None:
        st.error("Tidak dapat memuat data. Pastikan file data tersedia.")
        return
    
    # Ringkasan Supplier
    st.subheader("📈 Ringkasan Kinerja Supplier")
    
    # Buat dataframe ringkasan supplier
    supplier_cols = ["Supplier_ID"]
    if "Supplier_Name" in merged_df.columns:
        supplier_cols.append("Supplier_Name")
    if "Category" in merged_df.columns:
        supplier_cols.append("Category")
    if "Region" in merged_df.columns:
        supplier_cols.append("Region")
    
    supplier_stats_df = merged_df.groupby(supplier_cols).agg({
        "Lead_Time_Days": "mean",
        "Is_Late": "mean",
        "PO_ID": "count",
        "Quantity_Ordered": "sum"
    }).round(3).reset_index()
    
    if "Defect_Rate" in merged_df.columns:
        defect_stats = merged_df.groupby("Supplier_ID")["Defect_Rate"].mean().reset_index()
        supplier_stats_df = supplier_stats_df.merge(defect_stats, on="Supplier_ID", how="left")
    
    # Tampilkan tabel supplier
    st.dataframe(supplier_stats_df, use_container_width=True)
    
    st.markdown("---")
    
    # Prediksi PO Baru
    st.subheader("🔮 Prediksi Kinerja PO Baru")

    models = load_ml_models()
    supplier_stats_dict = calculate_supplier_stats(merged_df) if merged_df is not None else {}
    supplier_master_dict = create_supplier_master_dict(supplier_master)

    if models is None or merged_df is None:
        st.warning("⚠️ Model tidak tersedia untuk prediksi")
    else:
        with st.form("po_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                supp_id = st.selectbox("Pilih Supplier", options=sorted(merged_df["Supplier_ID"].unique()), key="po_supplier")
                po_quantity = st.number_input("Quantity PO", min_value=1, value=500)
                po_date = st.date_input("Tanggal PO", datetime.date.today(), key="po_date")
            
            with col2:
                # ✅ GANTI: hapus expected_lt_input, hanya pakai tanggal
                exp_delivery = st.date_input(
                    "Tanggal Pengiriman Diharapkan", 
                    value=po_date + datetime.timedelta(days=14)
                )
            
            predict_po_btn = st.form_submit_button("Prediksi Kinerja PO")
            
            if predict_po_btn:
                input_data = {
                    "Supplier_ID": supp_id,
                    "Order_Date": pd.to_datetime(po_date),
                    "Expected_Delivery_Date": pd.to_datetime(exp_delivery),
                    "Quantity_Ordered": po_quantity
                }
                
                predicted_lt = predict_lead_time(input_data, models, supplier_master_dict, supplier_stats_dict)
                prob_late = predict_is_late(input_data, models, supplier_master_dict, supplier_stats_dict)
                
                if predicted_lt is not None and prob_late is not None:
                    expected_lt = (exp_delivery - po_date).days
                    rule_late = predicted_lt > expected_lt
                    is_late_status = "Late" if prob_late > 0.5 else "On Time"
                    status_color = "red" if is_late_status == "Late" else "green"
                    
                    # --- Tampilan Utama ---
                    st.subheader("📊 Hasil Prediksi")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prediksi Lead Time", f"{predicted_lt:.1f} hari")
                    with col2:
                        st.metric("Janji Supplier", f"{expected_lt} hari")
                    with col3:
                        st.markdown(f"**Status (Model):** <span style='color:{status_color}'>{is_late_status}</span>", unsafe_allow_html=True)
                    
                    st.write(f"**Probabilitas Terlambat (Model):** {prob_late:.1%}")
                    
                    # --- Analisis Pendukung ---
                    st.markdown("---")
                    st.subheader("🔍 Analisis Pendukung")
                    diff = predicted_lt - expected_lt
                    if diff > 0:
                        st.info(f"📌 Prediksi lead time **{diff:.1f} hari lebih lama** dari janji supplier.")
                    elif diff < 0:
                        st.info(f"📌 Prediksi lead time **{abs(diff):.1f} hari lebih cepat** dari janji supplier.")
                    else:
                        st.info("📌 Prediksi lead time sesuai dengan janji supplier.")
                    
                    # Konsistensi model vs rule
                    if (prob_late > 0.5) == rule_late:
                        st.success("✅ Konsistensi tinggi: Model dan logika perbandingan lead time sejalan.")
                    else:
                        st.warning("⚠️ Peringatan: Model memprediksi 'Late', tetapi lead time prediksi ≤ janji (atau sebaliknya). Perlu audit fitur atau data.")
                    
                    # --- Rekomendasi ---
                    st.markdown("---")
                    st.subheader("💡 Rekomendasi")
                    if prob_late > 0.7:
                        st.error("🚨 Risiko tinggi keterlambatan. Pertimbangkan supplier alternatif atau safety stock tambahan.")
                    elif prob_late > 0.4:
                        st.warning("⚠️ Risiko sedang. Lakukan follow-up proaktif dengan supplier.")
                    else:
                        st.success("✅ Risiko rendah. Lanjutkan PO sesuai rencana.")
    st.markdown(
                """
                <hr style='margin-top: 50px;'>
                <p style='text-align: center; color: gray;'>
                    © 2025 Andre Nugraha. All rights reserved.
                </p>
                """,
                unsafe_allow_html=True
            )
# =============================================
# MAIN APP
# =============================================
def main():
    st.set_page_config(
        page_title="Supplier Performance Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar Navigation
    with st.sidebar:
        st.title("🧭 Navigation")
        page = st.radio(
            "Pilih Halaman:",
            ["Dashboard Utama", "Analisis Lead Time", "Analisis Supplier & Prediksi PO"]
        )
        
        st.subheader("ℹ️ About")
        st.info(
            "Dashboard ini menggunakan model ML untuk prediksi lead time dan keterlambatan PO. "
            "Semua prediksi menggunakan model yang sudah ditraining, bukan statistik sederhana."
        )
    
    # Routing halaman
    if page == "Dashboard Utama":
        page_dashboard()
    elif page == "Analisis Lead Time":
        page_lead_time_analysis()
    elif page == "Analisis Supplier & Prediksi PO":
        page_supplier_analysis()

if __name__ == "__main__":
    main()