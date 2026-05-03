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
# MULTI-LANGUAGE SUPPORT
# =============================================

TEXTS = {
    # Navigation
    "nav_title": {"id": "🧭 Navigasi", "en": "🧭 Navigation"},
    "page_dashboard": {"id": "Dashboard Utama", "en": "Main Dashboard"},
    "page_leadtime": {"id": "Analisis Lead Time", "en": "Lead Time Analysis"},
    "page_supplier": {"id": "Analisis Supplier & Prediksi PO", "en": "Supplier Analysis & PO Prediction"},
    "about": {"id": "ℹ️ Tentang", "en": "ℹ️ About"},
    "about_text": {"id": "Dashboard ini menggunakan model ML untuk prediksi lead time dan keterlambatan PO. Semua prediksi menggunakan model yang sudah ditraining, bukan statistik sederhana.", 
                   "en": "This dashboard uses ML models for lead time and PO delay predictions. All predictions use trained models, not simple statistics."},
    
    # Dashboard Page
    "dashboard_title": {"id": "🏠 Dashboard Utama - Supplier Performance Analytics", "en": "🏠 Main Dashboard - Supplier Performance Analytics"},
    "avg_lead_time": {"id": "Rata-rata Lead Time", "en": "Average Lead Time"},
    "days": {"id": "hari", "en": "days"},
    "late_rate": {"id": "Tingkat Keterlambatan", "en": "Late Rate"},
    "otif": {"id": "OTIF", "en": "OTIF"},
    "total_suppliers": {"id": "Total Supplier", "en": "Total Suppliers"},
    "trend_chart_title": {"id": "Trend Lead Time vs Tingkat Keterlambatan", "en": "Lead Time vs Late Rate Trend"},
    "month": {"id": "Bulan", "en": "Month"},
    "lead_time_days": {"id": "Lead Time (hari)", "en": "Lead Time (days)"},
    "late_rate_pct": {"id": "Late Rate (%)", "en": "Late Rate (%)"},
    "top_late_suppliers": {"id": "Top 5 Supplier Terlambat", "en": "Top 5 Late Suppliers"},
    "supplier_performance_title": {"id": "📊 Analisis Performa Supplier", "en": "📊 Supplier Performance Analysis"},
    "supplier_scatter_title": {"id": "Performa Supplier: Lead Time vs Tingkat Keterlambatan", "en": "Supplier Performance: Lead Time vs Late Rate"},
    "avg_lead_time_days": {"id": "Rata-rata Lead Time (hari)", "en": "Average Lead Time (days)"},
    "late_rate_value": {"id": "Tingkat Keterlambatan", "en": "Late Rate"},
    
    # Lead Time Analysis Page
    "leadtime_title": {"id": "⏱️ Analisis Lead Time", "en": "⏱️ Lead Time Analysis"},
    "avg_lt": {"id": "Rata-rata LT", "en": "Average LT"},
    "std_dev_lt": {"id": "Std Dev LT", "en": "Std Dev LT"},
    "fastest_lt": {"id": "LT Tercepat", "en": "Fastest LT"},
    "slowest_lt": {"id": "LT Terlama", "en": "Slowest LT"},
    "lt_distribution": {"id": "Distribusi Lead Time", "en": "Lead Time Distribution"},
    "lt_by_category": {"id": "Lead Time per Kategori", "en": "Lead Time by Category"},
    "lt_by_supplier": {"id": "Lead Time per Supplier", "en": "Lead Time by Supplier"},
    "top10_slowest": {"id": "Top 10 Supplier dengan Lead Time Terlama", "en": "Top 10 Suppliers with Slowest Lead Time"},
    "predict_lt": {"id": "🔮 Prediksi Lead Time", "en": "🔮 Predict Lead Time"},
    "supplier_id": {"id": "Supplier ID", "en": "Supplier ID"},
    "po_date": {"id": "Tanggal PO", "en": "PO Date"},
    "expected_delivery": {"id": "Tanggal Pengiriman Diharapkan", "en": "Expected Delivery Date"},
    "quantity": {"id": "Quantity", "en": "Quantity"},
    "predict_button": {"id": "Prediksi Lead Time", "en": "Predict Lead Time"},
    "predicted_lt": {"id": "Prediksi Lead Time", "en": "Predicted Lead Time"},
    "days_slower": {"id": "hari lebih lama", "en": "days slower"},
    "days_faster": {"id": "hari lebih cepat", "en": "days faster"},
    "exact_match": {"id": "sama persis", "en": "exact match"},
    "from_promise": {"id": "dari janji supplier", "en": "from supplier promise"},
    "historical_avg": {"id": "Rata-rata historis supplier", "en": "Historical average for supplier"},
    
    # Supplier Analysis Page
    "supplier_title": {"id": "🏭 Analisis Supplier & Prediksi PO", "en": "🏭 Supplier Analysis & PO Prediction"},
    "supplier_summary": {"id": "📈 Ringkasan Kinerja Supplier", "en": "📈 Supplier Performance Summary"},
    "predict_po": {"id": "🔮 Prediksi Kinerja PO Baru", "en": "🔮 New PO Performance Prediction"},
    "select_supplier": {"id": "Pilih Supplier", "en": "Select Supplier"},
    "po_quantity": {"id": "Quantity PO", "en": "PO Quantity"},
    "predict_po_button": {"id": "Prediksi Kinerja PO", "en": "Predict PO Performance"},
    "prediction_results": {"id": "📊 Hasil Prediksi", "en": "📊 Prediction Results"},
    "supplier_promise": {"id": "Janji Supplier", "en": "Supplier Promise"},
    "status_model": {"id": "Status (Model)", "en": "Status (Model)"},
    "late_probability": {"id": "Probabilitas Terlambat (Model)", "en": "Late Probability (Model)"},
    "supporting_analysis": {"id": "🔍 Analisis Pendukung", "en": "🔍 Supporting Analysis"},
    "consistency_high": {"id": "✅ Konsistensi tinggi: Model dan logika perbandingan lead time sejalan.",
                         "en": "✅ High consistency: Model and lead time comparison logic align."},
    "consistency_warning": {"id": "⚠️ Peringatan: Model memprediksi 'Late', tetapi lead time prediksi ≤ janji (atau sebaliknya). Perlu audit fitur atau data.",
                           "en": "⚠️ Warning: Model predicts 'Late', but predicted lead time ≤ promised (or vice versa). Feature or data audit needed."},
    "recommendations": {"id": "💡 Rekomendasi", "en": "💡 Recommendations"},
    "high_risk": {"id": "🚨 Risiko tinggi keterlambatan. Pertimbangkan supplier alternatif atau safety stock tambahan.",
                  "en": "🚨 High delay risk. Consider alternative suppliers or additional safety stock."},
    "medium_risk": {"id": "⚠️ Risiko sedang. Lakukan follow-up proaktif dengan supplier.",
                    "en": "⚠️ Medium risk. Proactively follow up with supplier."},
    "low_risk": {"id": "✅ Risiko rendah. Lanjutkan PO sesuai rencana.",
                 "en": "✅ Low risk. Proceed with PO as planned."},
    "late": {"id": "Terlambat", "en": "Late"},
    "on_time": {"id": "Tepat Waktu", "en": "On Time"},
    "warning_missing_qty": {"id": "⚠️ Kolom Quantity_Received tidak ditemukan. OTIF dihitung sebagai On-Time Rate saja.",
                           "en": "⚠️ Quantity_Received column not found. OTIF calculated as On-Time Rate only."},
    
    # Errors
    "error_load_data": {"id": "❌ Gagal memuat data", "en": "❌ Failed to load data"},
    "error_model_unavailable": {"id": "⚠️ Model tidak tersedia untuk prediksi", "en": "⚠️ Model not available for prediction"},
    "error_cannot_load": {"id": "Tidak dapat memuat data. Pastikan file data tersedia.", "en": "Cannot load data. Please ensure data files are available."},
    
    # Footer
    "copyright": {"id": "© 2025 Andre Nugraha. All rights reserved.", "en": "© 2025 Andre Nugraha. All rights reserved."}
}

def t(key, lang):
    """Get text in specified language"""
    return TEXTS.get(key, {}).get(lang, TEXTS.get(key, {}).get("id", key))

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
        supplier_master = pd.read_csv(DATA_DIR / "supplier_master.csv")
        
        po_df = pd.read_csv(DATA_DIR / "PO.csv", parse_dates=["Order_Date", "Expected_Delivery_Date"])
    
        gr_df = pd.read_csv(DATA_DIR / "GR.csv", parse_dates=["Actual_Delivery_Date"])
        
        return supplier_master, po_df, gr_df
    except Exception as e:
        st.error(f"❌ Gagal memuat data: {e}")
        return None, None, None

@st.cache_resource
def load_ml_models():
    try:
        lt_model    = joblib.load(MODELS_DIR / "leadtime_model.pkl")
        lt_features = joblib.load(FEATURES_DIR / "selected_features_LT.pkl")

        lt_te_encoder = None
        for fname in ["lt_supplier_te_encoder.pkl", "supplier_te_encoder.pkl", "te_encoder_LT.pkl"]:
            for ddir in [MODELS_DIR, FEATURES_DIR]:
                fpath = ddir / fname
                if fpath.exists():
                    lt_te_encoder = joblib.load(fpath)
                    break
            if lt_te_encoder is not None:
                break
 
        is_late_model = joblib.load(MODELS_DIR / "IsLate_model.pkl")
        with open(FEATURES_DIR / "selected_features_IsLate.json") as f:
            is_late_features = json.load(f)
 
        return {
            "leadtime": {
                "model":      lt_model,
                "features":   lt_features,
                "te_encoder": lt_te_encoder
            },
            "is_late": {
                "model":    is_late_model,
                "features": is_late_features
            }
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
    
    merged_df = pd.merge(po_df, gr_df, on="PO_ID", how="inner")
    
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
        Supplier_Std_LT=("Lead_Time_Days", "std"),
        Supplier_Late_Rate=("Is_Late", "mean"),
        Supplier_Late_Severity=("Delay_Days", "mean"),
        Supplier_Defect_Rate=("Defect_Rate", "mean"),
        Supplier_Reliability=("Is_Late", lambda x: 1 - x.mean()),
        Supplier_Order_Freq=("PO_ID", "count"),
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

def prepare_leadtime_features(input_data, supplier_master_dict, supplier_stats, te_encoder=None, lt_features=None):
    """Persiapkan fitur untuk model Lead Time."""
    supp_id = input_data["Supplier_ID"]
    
    if supp_id in supplier_master_dict:
        info = supplier_master_dict[supp_id]
        base_price = float(info["Base_Price"])
        category = str(info["Category"])
        region = str(info["Region"])
    else:
        base_price, category, region = 50.0, "Other", "Other"
    
    order_date_ts = pd.Timestamp(input_data["Order_Date"])
    exp_del_ts = pd.Timestamp(input_data["Expected_Delivery_Date"])
    expected_lt = (exp_del_ts - order_date_ts).days
    
    # Statistik supplier
    if supp_id in supplier_stats:
        s = supplier_stats[supp_id]
        supplier_avg_lt = s.get("Supplier_Avg_LT", 10.0)
        supplier_late_rate = s.get("Supplier_Late_Rate", 0.1)
        supplier_defect_rate = s.get("Supplier_Defect_Rate", 0.0)
        supplier_reliability = s.get("Supplier_Reliability", 1.0)
        supplier_late_sev = s.get("Supplier_Late_Severity", 0.0)
        supplier_std_lt = s.get("Supplier_Std_LT", 0.0)
        supplier_order_freq = s.get("Supplier_Order_Freq", 1.0)
    else:
        supplier_avg_lt, supplier_late_rate = 10.0, 0.1
        supplier_defect_rate, supplier_reliability = 0.0, 1.0
        supplier_late_sev, supplier_std_lt = 0.0, 0.0
        supplier_order_freq = 1.0
    
    row = {
        "Quantity_Ordered": float(input_data.get("Quantity_Ordered", 1)),
        "Base_Price": base_price,
        "Category": category,
        "Region": region,
        "Expected_Lead_Time": expected_lt,
        "Order_Year": order_date_ts.year,
        "Order_Month": order_date_ts.month,
        "Order_Quarter": order_date_ts.quarter,
        "Supplier_Avg_LT": supplier_avg_lt,
        "Supplier_Late_Rate": supplier_late_rate,
        "Supplier_Defect_Rate": supplier_defect_rate,
        "Supplier_Reliability": supplier_reliability,
        "Supplier_Late_Severity": supplier_late_sev,
        "Supplier_Std_LT": supplier_std_lt,
        "Supplier_Order_Freq": supplier_order_freq,
    }
    
    df = pd.DataFrame([row])
    
    if te_encoder is not None:
        try:
            supp_df = pd.DataFrame({"Supplier_ID": [supp_id]})
            te_val = te_encoder.transform(supp_df)[0, 0]
        except Exception:
            te_val = supplier_avg_lt
    else:
        te_val = supplier_avg_lt
    
    df["Supplier_ID_TE"] = float(te_val)    
    return df

def prepare_is_late_features(input_data, supplier_master_dict, supplier_stats):
    """Persiapkan fitur untuk model Is Late"""
    df = pd.DataFrame([input_data])
    supp_id = input_data["Supplier_ID"]
    
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
    df["Order_Quarter"] = df["Order_Date"].dt.quarter
    df["Order_Year"] = df["Order_Date"].dt.year
    
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
    """Prediksi lead time menggunakan model ML."""
    try:
        lt_model = models["leadtime"]["model"]
        te_encoder = models["leadtime"].get("te_encoder")
             
        lt_features = None
        
        if hasattr(lt_model, 'feature_names_in_'):
            lt_features = list(lt_model.feature_names_in_)
        
        elif models["leadtime"].get("features") is not None:
            lt_features = models["leadtime"]["features"]
            st.write(f"✅ Feature names dari file: {lt_features}")
        
        if lt_features is None and hasattr(lt_model, 'named_steps'):
            for step_name, step in lt_model.named_steps.items():
                if hasattr(step, 'get_feature_names_out'):
                    try:
                        sample_df = prepare_leadtime_features(
                            input_data, supplier_master_dict, supplier_stats, te_encoder
                        )
                        lt_features = list(step.get_feature_names_out())
                        st.write(f"✅ Feature names dari transformer: {lt_features}")
                        break
                    except:
                        pass
        
        
        df_features = prepare_leadtime_features(
            input_data, supplier_master_dict, supplier_stats, te_encoder
        )
        
        if 'Supplier_ID' in lt_features and 'Supplier_ID' not in df_features.columns:
            df_features['Supplier_ID'] = input_data['Supplier_ID']
        
        
        X = df_features[lt_features]
        
        predicted_lt = lt_model.predict(X)[0]
        
        return max(0, float(predicted_lt))
        
    except Exception as e:
        st.error(f"❌ Error dalam prediksi lead time: {e}")
        import traceback
        st.caption(f"🔍 Traceback: {traceback.format_exc()}")
        return None

def predict_is_late(input_data, models, supplier_master_dict, supplier_stats):
    """Prediksi keterlambatan menggunakan model ML.

    Model IsLate adalah self-contained pipeline (preprocessor ada di dalam),
    sehingga menerima raw DataFrame dengan kolom asli (termasuk Supplier_ID
    sebagai string, Category, Region apa adanya).
    """
    try:
        df_features      = prepare_is_late_features(input_data, supplier_master_dict, supplier_stats)
        is_late_features = models["is_late"]["features"]
        is_late_model    = models["is_late"]["model"]

        X = df_features[is_late_features]
        prob_late = is_late_model.predict_proba(X)[0][1]

        return float(prob_late)
    except Exception as e:
        st.error(f"❌ Error dalam prediksi keterlambatan: {e}")
        import traceback
        st.caption(f"🔍 Traceback: {traceback.format_exc()}")
        return None

# =============================================
# HALAMAN 1: DASHBOARD UTAMA
# =============================================
def page_dashboard(lang_code):
    st.title(t("dashboard_title", lang_code))
    
    supplier_master, merged_df, po_df = load_processed_data()
    
    if merged_df is None:
        st.error(t("error_cannot_load", lang_code))
        return
    
    if "Quantity_Received" in merged_df.columns and "Quantity_Ordered" in merged_df.columns:
        merged_df["In_Full"] = merged_df["Quantity_Received"] == merged_df["Quantity_Ordered"]
        merged_df["OTIF"] = (~merged_df["Is_Late"]) & merged_df["In_Full"]
        otif_rate = merged_df["OTIF"].mean() * 100
    else:
        otif_rate = (1 - merged_df["Is_Late"].mean()) * 100
        st.warning(t("warning_missing_qty", lang_code))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_lt = merged_df["Lead_Time_Days"].mean()
        st.metric(t("avg_lead_time", lang_code), f"{avg_lt:.1f} {t('days', lang_code)}")

    with col2:
        late_rate = merged_df["Is_Late"].mean() * 100
        st.metric(t("late_rate", lang_code), f"{late_rate:.1f}%")

    with col3:
        st.metric(t("otif", lang_code), f"{otif_rate:.1f}%")

    with col4:
        total_suppliers = merged_df["Supplier_ID"].nunique()
        st.metric(t("total_suppliers", lang_code), f"{total_suppliers}")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        monthly_lt = merged_df.groupby(merged_df["Order_Date"].dt.to_period("M")).agg({
            "Lead_Time_Days": "mean",
            "Is_Late": "mean"
        }).reset_index()
        monthly_lt["Order_Date"] = monthly_lt["Order_Date"].dt.to_timestamp()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=monthly_lt["Order_Date"], y=monthly_lt["Lead_Time_Days"], 
                      name=t("avg_lead_time", lang_code), line=dict(color="blue")),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=monthly_lt["Order_Date"], y=monthly_lt["Is_Late"]*100, 
                      name=t("late_rate", lang_code), line=dict(color="red")),
            secondary_y=True,
        )
        
        fig.update_layout(
            title=t("trend_chart_title", lang_code),
            xaxis_title=t("month", lang_code),
            hovermode="x unified"
        )
        
        fig.update_yaxes(title_text=t("lead_time_days", lang_code), secondary_y=False)
        fig.update_yaxes(title_text=t("late_rate_pct", lang_code), secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(t("top_late_suppliers", lang_code))
        late_suppliers = merged_df.groupby("Supplier_ID")["Is_Late"].mean().sort_values(ascending=False).head(5)
        for supp, rate in late_suppliers.items():
            st.write(f"**{supp}**: {rate:.1%}")
    
    st.subheader(t("supplier_performance_title", lang_code))
    supplier_perf = merged_df.groupby("Supplier_ID").agg({
        "Lead_Time_Days": "mean",
        "Is_Late": "mean",
        "PO_ID": "count"
    }).round(3).reset_index()
    
    fig = px.scatter(supplier_perf, x="Lead_Time_Days", y="Is_Late", 
                     size="PO_ID", color="Is_Late",
                     hover_data=["Supplier_ID"],
                     title=t("supplier_scatter_title", lang_code),
                     labels={"Lead_Time_Days": t("avg_lead_time_days", lang_code), 
                            "Is_Late": t("late_rate_value", lang_code)})
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
                f"""
                <hr style='margin-top: 50px;'>
                <p style='text-align: center; color: gray;'>
                    {t("copyright", lang_code)}
                </p>
                """,
                unsafe_allow_html=True
            )

# =============================================
# HALAMAN 2: ANALISIS LEAD TIME
# =============================================
def page_lead_time_analysis(lang_code):
    st.title(t("leadtime_title", lang_code))
    
    supplier_master, merged_df, po_df = load_processed_data()
    
    if merged_df is None:
        st.error(t("error_cannot_load", lang_code))
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_lt = merged_df["Lead_Time_Days"].mean()
        st.metric(t("avg_lt", lang_code), f"{avg_lt:.1f} {t('days', lang_code)}")
    
    with col2:
        std_lt = merged_df["Lead_Time_Days"].std()
        st.metric(t("std_dev_lt", lang_code), f"{std_lt:.1f} {t('days', lang_code)}")
    
    with col3:
        min_lt = merged_df["Lead_Time_Days"].min()
        st.metric(t("fastest_lt", lang_code), f"{min_lt} {t('days', lang_code)}")
    
    with col4:
        max_lt = merged_df["Lead_Time_Days"].max()
        st.metric(t("slowest_lt", lang_code), f"{max_lt} {t('days', lang_code)}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t("lt_distribution", lang_code))
        fig = px.histogram(merged_df, x="Lead_Time_Days", 
                          title=t("lt_distribution", lang_code),
                          nbins=20,
                          labels={"Lead_Time_Days": t("lead_time_days", lang_code)})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if "Category" in merged_df.columns:
            st.subheader(t("lt_by_category", lang_code))
            fig = px.box(merged_df, x="Category", y="Lead_Time_Days",
                        title=t("lt_by_category", lang_code),
                        labels={"Lead_Time_Days": t("lead_time_days", lang_code), "Category": "Category"})
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader(t("lt_by_supplier", lang_code))
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
                    title=t("top10_slowest", lang_code),
                    color="Avg_LT",
                    labels={"Avg_LT": t("avg_lead_time_days", lang_code)})
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader(t("predict_lt", lang_code))

    models = load_ml_models()
    supplier_stats = calculate_supplier_stats(merged_df) if merged_df is not None else {}
    supplier_master_dict = create_supplier_master_dict(supplier_master)

    if models is None or merged_df is None:
        st.warning(t("error_model_unavailable", lang_code))
    else:
        with st.form("lt_prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                supplier_id = st.selectbox(t("supplier_id", lang_code), options=sorted(merged_df["Supplier_ID"].unique()))
            
            with col2:
                order_date = st.date_input(t("po_date", lang_code), datetime.date.today())
                expected_delivery_date = st.date_input(
                    t("expected_delivery", lang_code), 
                    value=order_date + datetime.timedelta(days=14)
                )
            
            with col3:
                quantity = st.number_input(t("quantity", lang_code), min_value=1, value=100)
            
            predict_btn = st.form_submit_button(t("predict_button", lang_code))
            
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
                    
                    st.success(f"**{t('predicted_lt', lang_code)}: {predicted_lt:.1f} {t('days', lang_code)}**")
                    
                    if diff > 0:
                        st.info(f"📦 Prediksi **{diff:.1f} {t('days_slower', lang_code)}** {t('from_promise', lang_code)} ({expected_lt} {t('days', lang_code)}).")
                    elif diff < 0:
                        st.info(f"🚀 Prediksi **{abs(diff):.1f} {t('days_faster', lang_code)}** {t('from_promise', lang_code)} ({expected_lt} {t('days', lang_code)}).")
                    else:
                        st.info(f"✅ Prediksi **{t('exact_match', lang_code)}** {t('from_promise', lang_code)} ({expected_lt} {t('days', lang_code)}).")
                    
                    if supplier_id in supplier_stats:
                        hist_avg = supplier_stats[supplier_id]["Supplier_Avg_LT"]
                        st.write(f"📈 **{t('historical_avg', lang_code)} {supplier_id}: {hist_avg:.1f} {t('days', lang_code)}**")

# =============================================
# HALAMAN 3: ANALISIS SUPPLIER & PREDIKSI PO
# =============================================
def page_supplier_analysis(lang_code):
    st.title(t("supplier_title", lang_code))
    
    supplier_master, merged_df, po_df = load_processed_data()
    
    if merged_df is None:
        st.error(t("error_cannot_load", lang_code))
        return
    
    st.subheader(t("supplier_summary", lang_code))
    
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
    
    st.dataframe(supplier_stats_df, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader(t("predict_po", lang_code))

    models = load_ml_models()
    supplier_stats_dict = calculate_supplier_stats(merged_df) if merged_df is not None else {}
    supplier_master_dict = create_supplier_master_dict(supplier_master)

    if models is None or merged_df is None:
        st.warning(t("error_model_unavailable", lang_code))
    else:
        with st.form("po_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                supp_id = st.selectbox(t("select_supplier", lang_code), options=sorted(merged_df["Supplier_ID"].unique()), key="po_supplier")
                po_quantity = st.number_input(t("po_quantity", lang_code), min_value=1, value=500)
                po_date = st.date_input(t("po_date", lang_code), datetime.date.today(), key="po_date")
            
            with col2:
                exp_delivery = st.date_input(
                    t("expected_delivery", lang_code), 
                    value=po_date + datetime.timedelta(days=14)
                )
            
            predict_po_btn = st.form_submit_button(t("predict_po_button", lang_code))
            
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
                    is_late_status = t("late", lang_code) if prob_late > 0.5 else t("on_time", lang_code)
                    status_color = "red" if is_late_status == t("late", lang_code) else "green"
                    
                    st.subheader(t("prediction_results", lang_code))
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(t("predicted_lt", lang_code), f"{predicted_lt:.1f} {t('days', lang_code)}")
                    with col2:
                        st.metric(t("supplier_promise", lang_code), f"{expected_lt} {t('days', lang_code)}")
                    with col3:
                        st.markdown(f"**{t('status_model', lang_code)}:** <span style='color:{status_color}'>{is_late_status}</span>", unsafe_allow_html=True)
                    
                    st.write(f"**{t('late_probability', lang_code)}:** {prob_late:.1%}")
                    
                    st.markdown("---")
                    st.subheader(t("supporting_analysis", lang_code))
                    diff = predicted_lt - expected_lt
                    if diff > 0:
                        st.info(f"📌 Prediksi lead time **{diff:.1f} {t('days', lang_code)} {t('days_slower', lang_code)}** {t('from_promise', lang_code)}.")
                    elif diff < 0:
                        st.info(f"📌 Prediksi lead time **{abs(diff):.1f} {t('days', lang_code)} {t('days_faster', lang_code)}** {t('from_promise', lang_code)}.")
                    else:
                        st.info(f"📌 Prediksi lead time {t('exact_match', lang_code)} {t('from_promise', lang_code)}.")
                    
                    if (prob_late > 0.5) == rule_late:
                        st.success(t("consistency_high", lang_code))
                    else:
                        st.warning(t("consistency_warning", lang_code))
                    
                    st.markdown("---")
                    st.subheader(t("recommendations", lang_code))
                    if prob_late > 0.7:
                        st.error(t("high_risk", lang_code))
                    elif prob_late > 0.4:
                        st.warning(t("medium_risk", lang_code))
                    else:
                        st.success(t("low_risk", lang_code))
    
    st.markdown(
                f"""
                <hr style='margin-top: 50px;'>
                <p style='text-align: center; color: gray;'>
                    {t("copyright", lang_code)}
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
    
    # Initialize session state
    if 'lang_code' not in st.session_state:
        st.session_state.lang_code = "id"
    
    # === LANGUAGE TOGGLE DI PALING ATAS (SEBELUM JUDUL) ===
    # Gunakan container kosong untuk menggeser toggle ke kanan atas
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col2:
        # Indicator bahasa saat ini
        if st.session_state.lang_code == "id":
            st.markdown("**🇮🇩 ID**")
        else:
            st.markdown("**🇬🇧 EN**")
    
    with col3:
        # Native Streamlit toggle switch
        is_english = st.toggle(
            "🌐", 
            value=(st.session_state.lang_code == "en"),
            key="lang_toggle",
            label_visibility="collapsed"
        )
    
    # Update language based on toggle
    new_lang_code = "en" if is_english else "id"
    if new_lang_code != st.session_state.lang_code:
        st.session_state.lang_code = new_lang_code
        st.rerun()
    
    lang_code = st.session_state.lang_code
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("---")
        page = st.radio(
            "Pilih Halaman:" if lang_code == "id" else "Select Page:",
            [
                t("page_dashboard", lang_code), 
                t("page_leadtime", lang_code), 
                t("page_supplier", lang_code)
            ]
        )
        
        st.markdown("---")
        
        if lang_code == "id":
            st.info("🌐 Bahasa: Indonesia")
        else:
            st.info("🌐 Language: English")
        
        st.subheader(t("about", lang_code))
        st.info(t("about_text", lang_code))
    
    # Page Routing - Page functions TANPA st.title
    if page == t("page_dashboard", lang_code):
        page_dashboard(lang_code)
    elif page == t("page_leadtime", lang_code):
        page_lead_time_analysis(lang_code)
    elif page == t("page_supplier", lang_code):
        page_supplier_analysis(lang_code)

if __name__ == "__main__":
    main()