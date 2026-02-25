import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing 
from sklearn.ensemble import RandomForestClassifier 
import warnings
warnings.filterwarnings('ignore')

# Import Modul Buatan Sendiri
from auth import check_password
from data_engine import load_data_hybrid

# --- 1. KONFIGURASI TAMPILAN ---
st.set_page_config(
    page_title="ACA BOGOR: EXECUTIVE DASHBOARD", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# CSS TAMPILAN
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 3rem; }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border-left: 5px solid #2563eb;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 700; color: #475569; }
</style>
""", unsafe_allow_html=True)

# WARNA TEMA
C_PRIM = "#2563eb"
C_DANG = "#ef4444" 
C_SUCC = "#22c55e" 
C_WARN = "#f59e0b"

def make_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0), font=dict(family="Arial", size=11, color="#334155"),
        xaxis=dict(showgrid=False, linecolor="#cbd5e1"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
    )
    return fig

# --- 2. SISTEM LOGIN ---
check_password() # Memanggil fungsi login dari auth.py

# --- 3. UI DASHBOARD UTAMA ---
col_logo, col_logout = st.sidebar.columns([3, 1])
if col_logout.button("Keluar"):
    st.session_state['logged_in'] = False
    st.rerun()

st.sidebar.header("CONTROL PANEL")
date_option = st.sidebar.radio("Format Tanggal:", ["AUTO (Deteksi)", "INDO (Hari/Bulan)", "US (Bulan/Hari)"], index=0)
uploaded_files = st.sidebar.file_uploader("Upload CSV/Parquet Tambahan", type=['csv', 'parquet'], accept_multiple_files=True)

# Memanggil mesin data dari data_engine.py
df_raw, error_msg = load_data_hybrid(uploaded_files, date_option)

if df_raw is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("FILTER DATA")
    
    min_date = df_raw['TGL_IN'].min().date()
    max_date = df_raw['TGL_IN'].max().date()
    
    date_range = st.sidebar.date_input("Pilih Rentang Waktu:", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df_raw[(df_raw['TGL_IN'].dt.date >= start_date) & (df_raw['TGL_IN'].dt.date <= end_date)]
    else:
        df = df_raw.copy()
        start_date, end_date = min_date, max_date
    
    all_segments = ["SEMUA"] + sorted(df['SEGMENT'].unique().tolist())
    sel_segment = st.sidebar.selectbox("Pilih Segmen Bisnis:", all_segments)
    if sel_segment != "SEMUA": df = df[df['SEGMENT'] == sel_segment]
    
    all_products = ["SEMUA"] + sorted(df['TOC_DESCRIPTION'].unique().tolist())
    sel_product = st.sidebar.selectbox("Pilih Produk:", all_products)
    if sel_product != "SEMUA": df = df[df['TOC_DESCRIPTION'] == sel_product]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("SIMULASI TARGET")
    target_pct = st.sidebar.slider("Asumsi Target Growth YoY (%)", min_value=5, max_value=100, value=15, step=1)
    st.sidebar.info(f"DATA TAMPIL: {len(df):,} Transaksi")
    
    # --- UI BODY ---
    st.title("DASHBOARD ACA BOGOR")
    if len(date_range) == 2:
        st.caption(f"Monitoring Data: {start_date.strftime('%d %b %Y')} s/d {end_date.strftime('%d %b %Y')} | Segmen: {sel_segment} | Produk: {sel_product}")
    
    if len(df) > 0:
        current_year = df['TGL_IN'].dt.year.max()
        curr_ytd_df = df[df['TGL_IN'].dt.year == current_year]
        prev_year_df = df_raw[(df_raw['TGL_IN'].dt.year == current_year - 1) & (df_raw['TGL_IN'].dt.month <= df['TGL_IN'].dt.month.max())]
        
        curr_omset = curr_ytd_df['PREMIUM'].sum()
        prev_omset = prev_year_df['PREMIUM'].sum()
        growth_yoy = ((curr_omset - prev_omset)/prev_omset * 100) if prev_omset > 0 else 0
        target_omset = prev_omset * (1 + (target_pct / 100))
        achievment = (curr_omset / target_omset * 100) if target_omset > 0 else 0
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"TOTAL OMSET YTD ({current_year})", f"{curr_omset/1e9:.2f} M", f"{growth_yoy:.1f}% vs Tahun Lalu")
        k2.metric("PENCAPAIAN TARGET", f"{achievment:.1f}%", f"Asumsi Target +{target_pct}%")
        k3.metric("KECEPATAN (SLA)", f"{df['SLA_HARI'].mean():.1f} HARI", "Target < 2")
        k4.metric("TOTAL POLIS", f"{len(df):,}", "Transaksi Aktif")

        with st.expander("Download Laporan (PDF / HTML)"):
            st.write("Klik tombol di bawah ini untuk mengunduh laporan ringkas.")
            top_prod_report = df.groupby('TOC_DESCRIPTION')['PREMIUM'].sum().nlargest(3).reset_index()
            prod_list_html = "".join([f"<li>{row['TOC_DESCRIPTION']} (Rp {row['PREMIUM']/1e6:,.0f} Juta)</li>" for _, row in top_prod_report.iterrows()])
            
            report_html = f"""
            <html><head><title>Laporan Dashboard ACA Bogor</title></head>
            <body style='font-family:Arial, sans-serif; padding:40px; color:#333;'>
                <h1 style='color:#2563eb; border-bottom: 2px solid #2563eb;'>LAPORAN DASHBOARD ACA BOGOR</h1>
                <p><b>Tanggal Download:</b> {datetime.now().strftime('%d %B %Y %H:%M')}</p>
                <p><b>Periode Data:</b> {start_date} s/d {end_date}</p>
                <ul>
                    <li><b>Total Omset Premium:</b> Rp {curr_omset:,.0f}</li>
                    <li><b>Total Transaksi:</b> {len(df):,} Polis</li>
                    <li><b>Pertumbuhan (YoY):</b> {growth_yoy:.2f}%</li>
                </ul>
                <h3>Top 3 Produk Penyumbang Omset</h3>
                <ul>{prod_list_html}</ul>
            </body></html>
            """
            st.download_button(label="Download Laporan", data=report_html, file_name=f"Report_{datetime.now().strftime('%Y%m%d')}.html", mime="text/html")

        st.markdown("---")
        t1, t2, t3, t4, t5, t6, t7 = st.tabs(["TREN BISNIS", "PRODUK", "NASABAH (CRM)", "AI INTELLIGENCE", "OPERASIONAL", "DATA", "MITRA (AGEN & BROKER)"])
        
        with t1:
            c1, c2 = st.columns([2,1])
            with c1:
                st.subheader("TREN PENDAPATAN")
                trend = df.groupby(['TAHUN', 'BULAN_NUM', 'BULAN_NAMA'])['PREMIUM'].sum().reset_index().sort_values(['TAHUN', 'BULAN_NUM'])
                fig = px.line(trend, x='BULAN_NAMA', y='PREMIUM', color='TAHUN', markers=True, color_discrete_sequence=px.colors.qualitative.Bold)
                st.plotly_chart(make_chart(fig), use_container_width=True)
            with c2:
                st.subheader("POLA MUSIMAN")
                seas = df.groupby('BULAN_NAMA')['PREMIUM'].mean().reindex(['JANUARY','FEBRUARY','MARCH','APRIL','MAY','JUNE','JULY','AUGUST','SEPTEMBER','OCTOBER','NOVEMBER','DECEMBER']).reset_index()
                fig_s = px.bar(seas, x='BULAN_NAMA', y='PREMIUM', color='PREMIUM')
                st.plotly_chart(make_chart(fig_s), use_container_width=True)
            
            st.markdown("---")
            st.subheader("AI FORECASTER")
            df_ts = trend.copy()
            df_ts['PERIODE'] = df_ts['TAHUN'].astype(str) + "-" + df_ts['BULAN_NUM'].astype(str).str.zfill(2)
            
            if len(df_ts) >= 6:
                future_steps = 6
                last_year = int(df_ts['TAHUN'].iloc[-1])
                last_month = int(df_ts['BULAN_NUM'].iloc[-1])
                pred_list = []
                
                if len(df_ts) >= 12:
                    try:
                        ts_data = df_ts['PREMIUM'].values
                        model = ExponentialSmoothing(ts_data, trend='add', seasonal='add', seasonal_periods=12, initialization_method="estimated").fit()
                        pred_vals = model.forecast(future_steps)
                        forecast_type = 'AI Forecast (Holt-Winters)'
                    except:
                        z = np.polyfit(np.arange(len(df_ts)), df_ts['PREMIUM'].values, 1)
                        pred_vals = [np.poly1d(z)(len(df_ts) - 1 + i) for i in range(1, future_steps + 1)]
                        forecast_type = 'AI Forecast (Tren Linear)'
                else:
                    z = np.polyfit(np.arange(len(df_ts)), df_ts['PREMIUM'].values, 1)
                    pred_vals = [np.poly1d(z)(len(df_ts) - 1 + i) for i in range(1, future_steps + 1)]
                    forecast_type = 'AI Forecast (Tren Linear)'

                for i, p_val in enumerate(pred_vals):
                    next_month = last_month + (i + 1)
                    next_year = last_year + (next_month - 1) // 12
                    next_month = (next_month - 1) % 12 + 1
                    pred_list.append({'PERIODE': f"{next_year}-{str(next_month).zfill(2)}", 'PREMIUM': max(p_val, 0), 'TIPE': forecast_type})
                
                df_ts['TIPE'] = 'Data Aktual'
                df_combined = pd.concat([df_ts[['PERIODE', 'PREMIUM', 'TIPE']].tail(12), pd.DataFrame(pred_list)])
                fig_cast = px.bar(df_combined, x='PERIODE', y='PREMIUM', color='TIPE', color_discrete_map={'Data Aktual': C_PRIM, forecast_type: C_WARN})
                st.plotly_chart(make_chart(fig_cast), use_container_width=True)
            else:
                st.info("Pilih rentang waktu minimal 6 bulan untuk mengaktifkan AI Forecaster.")

        with t2:
            st.subheader("ANALISIS PRODUK & PORTOFOLIO")
            c_p1, c_p2 = st.columns([1, 1]) 
            
            with c_p1:
                st.write("**PETA DOMINASI PRODUK**")
                df_pos = df[df['PREMIUM']>0]
                if not df_pos.empty:
                    fig_tree = px.treemap(df_pos, path=[px.Constant("TOTAL"), 'SEGMENT', 'TOC_DESCRIPTION'], values='PREMIUM', color='SEGMENT', color_discrete_sequence=px.colors.qualitative.Prism)
                    st.plotly_chart(make_chart(fig_tree), use_container_width=True)
            
            with c_p2:
                st.write("**TOP 10 PRODUK (BERDASARKAN PREMI)**")
                
                list_segmen = ["SEMUA SEGMEN"] + list(df['SEGMENT'].dropna().unique())
                pilihan_segmen = st.selectbox("Filter berdasarkan Segmen:", list_segmen)
                
                if pilihan_segmen == "SEMUA SEGMEN":
                    df_plot_base = df 
                else:
                    df_plot_base = df[df['SEGMENT'] == pilihan_segmen]
                
                if df_plot_base.empty:
                    st.info(f"Belum ada transaksi untuk segmen: {pilihan_segmen}.")
                else:
                    df_grouped = df_plot_base.groupby('TOC_DESCRIPTION')['PREMIUM'].sum().reset_index()
                    df_grouped = df_grouped.sort_values('PREMIUM', ascending=False)
                    
                    top_n = 10
                    if len(df_grouped) > top_n:
                        df_top = df_grouped.iloc[:top_n]
                        df_others = pd.DataFrame({
                            'TOC_DESCRIPTION': ['LAINNYA (GABUNGAN)'],
                            'PREMIUM': [df_grouped.iloc[top_n:]['PREMIUM'].sum()]
                        })
                        df_plot = pd.concat([df_top, df_others], ignore_index=True)
                    else:
                        df_plot = df_grouped

                    fig_bar = px.bar(
                        df_plot, 
                        x='PREMIUM', 
                        y='TOC_DESCRIPTION', 
                        orientation='h', 
                        text='PREMIUM', 
                        color_discrete_sequence=[C_PRIM]
                    )
                    
                    fig_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                    fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
                    
                    st.plotly_chart(make_chart(fig_bar), use_container_width=True)
                
        with t3:
            st.subheader("ANALISIS NASABAH & AI CHURN PREDICTION")
            snap_date = df['TGL_IN'].max() + timedelta(days=1)
            rfm = df.groupby('INSURED_NAME').agg({'TGL_IN': lambda x: (snap_date - x.max()).days, 'POLICYNO': 'count', 'PREMIUM': 'sum'}).rename(columns={'TGL_IN': 'RECENCY', 'POLICYNO': 'FREQ', 'PREMIUM': 'MONETARY'})
            rfm['STATUS'] = rfm.apply(lambda r: "CHAMPIONS (VIP)" if r['RECENCY'] < 180 and r['MONETARY'] > 100000000 else ("ACTIVE" if r['RECENCY'] < 180 else ("RISIKO CHURN" if r['RECENCY'] < 365 else "SLEEPING (DORMAN)")), axis=1)
            
            # --- START ML CHURN PREDICTION ---
            rfm['IS_CHURN_HIST'] = (rfm['RECENCY'] > 180).astype(int)
            if len(rfm['IS_CHURN_HIST'].unique()) > 1:
                X = rfm[['FREQ', 'MONETARY']]
                y = rfm['IS_CHURN_HIST']
                rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                rf_model.fit(X, y)
                rfm['PROB_CHURN_%'] = rf_model.predict_proba(X)[:, 1] * 100
            else:
                rfm['PROB_CHURN_%'] = 0.0
            # --- END ML CHURN PREDICTION ---

            c_rfm1, c_rfm2 = st.columns([1,2])
            with c_rfm1:
                fig_pie = px.pie(rfm['STATUS'].value_counts().reset_index(), values='count', names='STATUS', hole=0.5)
                st.plotly_chart(make_chart(fig_pie), use_container_width=True)
            with c_rfm2:
                st.write("**AI PREDICTION: NASABAH AKTIF YANG BERISIKO TINGGI CHURN**")
                at_risk = rfm[rfm['STATUS'] == 'ACTIVE'].sort_values('PROB_CHURN_%', ascending=False).head(10)
                if not at_risk.empty:
                    st.dataframe(
                        at_risk[['PROB_CHURN_%', 'FREQ', 'MONETARY']], 
                        use_container_width=True,
                        column_config={"PROB_CHURN_%": st.column_config.ProgressColumn("Probabilitas Churn", min_value=0, max_value=100, format="%.1f%%")}
                    )
                else:
                    st.success("Belum ada indikasi nasabah aktif yang berisiko churn tinggi saat ini.")
            
            st.markdown("---")
            st.write("**PETA SEBARAN NASABAH (DETAIL PER KLUSTER)**")
            
            rfm_plot = rfm.reset_index().copy()
            rfm_plot['MONETARY_SIZE'] = rfm_plot['MONETARY'].abs() + 1 
            
            c_fil1, c_fil2 = st.columns([1, 4])
            
            with c_fil1:
                st.caption("FILTER GRAFIK:")
                filter_status = st.radio(
                    "Pilih Kluster:", 
                    ["TAMPILKAN SEMUA", "CHAMPIONS (VIP)", "ACTIVE", "RISIKO CHURN", "SLEEPING (DORMAN)"]
                )
            
            if filter_status != "TAMPILKAN SEMUA":
                rfm_plot = rfm_plot[rfm_plot['STATUS'] == filter_status]
            
            with c_fil2:
                if rfm_plot.empty:
                    st.info(f"Belum ada data nasabah untuk kategori: {filter_status}.")
                else:
                    fig_2d = px.scatter(
                        rfm_plot, 
                        x='RECENCY', 
                        y='FREQ', 
                        size='MONETARY_SIZE', 
                        color='STATUS', 
                        hover_name='INSURED_NAME',
                        hover_data={'MONETARY': ':,.0f', 'MONETARY_SIZE': False},
                        opacity=0.7, 
                        color_discrete_map={
                            'CHAMPIONS (VIP)': C_SUCC, 
                            'ACTIVE': C_PRIM, 
                            'RISIKO CHURN': C_WARN, 
                            'SLEEPING (DORMAN)': C_DANG
                        },
                        labels={'RECENCY': 'Recency (Jarak Hari)', 'FREQ': 'Frequency (Jumlah Transaksi)'}
                    )
                    st.plotly_chart(make_chart(fig_2d), use_container_width=True)

        with t4:
            st.markdown("### AI INTELLIGENCE CENTER")
            
            st.info("**1. AI OPPORTUNITY FINDER (Peluang Cross-Selling)**")
            basket = df.groupby(['INSURED_NAME', 'TOC_DESCRIPTION'])['POLICYNO'].count().unstack().fillna(0).map(lambda x: 1 if x > 0 else 0)
            if len(basket.columns) > 1:
                co_matrix = basket.T.dot(basket)
                np.fill_diagonal(co_matrix.values, 0)
                
                stacked = co_matrix.stack()
                stacked.index.names = ['Product A', 'Product B'] 
                pairs = stacked.reset_index(name='Count')
                pairs = pairs[pairs['Count'] > 0]
                pairs['sorted_key'] = pairs.apply(lambda x: tuple(sorted([x['Product A'], x['Product B']])), axis=1)
                top_pairs = pairs.drop_duplicates(subset='sorted_key').sort_values('Count', ascending=False).head(10)
                
                if not top_pairs.empty:
                    top_pairs['Pair Name'] = top_pairs['Product A'] + " + " + top_pairs['Product B']
                    fig_pairs = px.bar(top_pairs, x='Count', y='Pair Name', orientation='h')
                    st.plotly_chart(make_chart(fig_pairs), use_container_width=True)
                else: 
                    st.warning("Belum cukup data cross-selling.")
            else: 
                st.warning("Variasi produk belum cukup.")

            st.markdown("---")
            
            # --- MULAI PERBAIKAN GRAFIK ANOMALI (LOG SCALE) ---
            st.error("**2. AI AUDITOR (Deteksi Anomali Rate)**")
            st.caption("Mendeteksi polis dengan Rate (%) yang terlalu tinggi atau rendah dibandingkan rata-rata produknya.")
            
            stats = df[df['RATE_PCT'] > 0].groupby('TOC_DESCRIPTION')['RATE_PCT'].agg(['mean', 'std']).reset_index()
            df_risk = pd.merge(df, stats, on='TOC_DESCRIPTION', how='left')
            df_risk['Z_SCORE'] = (df_risk['RATE_PCT'] - df_risk['mean']) / df_risk['std']
            
            # PERBAIKAN 1: Z-Score dibuat lebih ketat (> 3) agar tidak terlalu banyak mendeteksi anomali
            df_risk['IS_ANOMALY'] = np.where((df_risk['Z_SCORE'].abs() > 3) & (df_risk['PREMIUM'] > 1000000), 'Anomali', 'Normal')
            
            c_anom1, c_anom2 = st.columns([2, 1])
            
            with c_anom1:
                # Pastikan Premium di atas nol (karena logaritma dari 0 atau negatif itu error)
                df_plot_anom = df_risk[df_risk['PREMIUM'] > 0]
                
                fig_anom = px.scatter(
                    df_plot_anom, 
                    x="PREMIUM", 
                    y="RATE_PCT", 
                    color="IS_ANOMALY", 
                    color_discrete_map={'Anomali': C_DANG, 'Normal': 'rgba(203, 213, 225, 0.3)'}, 
                    hover_name="INSURED_NAME",
                    hover_data={
                        "POLICYNO": True, 
                        "TOC_DESCRIPTION": True, 
                        "RATE_PCT": ":.2f", 
                        "IS_ANOMALY": False,
                        "PREMIUM": ":,.0f"
                    },
                    labels={'PREMIUM': 'Nilai Premi (Rp) - Skala Log', 'RATE_PCT': 'Rate (%)'},
                    log_x=True # PERBAIKAN 2: Menggunakan Skala Logaritmik agar data tidak tergencet
                )
                
                # Mengganti format angka sumbu X agar mudah dibaca di mode Log
                fig_anom.update_layout(xaxis=dict(tickformat='.0s'))
                fig_anom.update_traces(marker=dict(size=6, line=dict(width=0))) 
                st.plotly_chart(make_chart(fig_anom), use_container_width=True)
                
            with c_anom2:
                anomalies_df = df_risk[df_risk['IS_ANOMALY'] == 'Anomali'].sort_values('Z_SCORE', key=abs, ascending=False)
                st.write(f"**🔴 {len(anomalies_df)} Polis Terdeteksi Anomali Ekstrem**")
                
                if not anomalies_df.empty:
                    st.dataframe(
                        anomalies_df[['POLICYNO', 'RATE_PCT']],
                        use_container_width=True,
                        hide_index=True,
                        column_config={"RATE_PCT": st.column_config.NumberColumn("Rate (%)", format="%.2f")}
                    )
                else:
                    st.success("Aman! Tidak ada indikasi anomali ekstrem pada rate premi saat ini.")
                    
            if not anomalies_df.empty:
                with st.expander("Buka Rincian Lengkap Data Anomali Ekstrem"):
                    st.dataframe(
                        anomalies_df[['POLICYNO', 'INSURED_NAME', 'TOC_DESCRIPTION', 'PREMIUM', 'RATE_PCT', 'mean', 'Z_SCORE']],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "POLICYNO": "No. Polis",
                            "INSURED_NAME": "Nama Nasabah",
                            "TOC_DESCRIPTION": "Produk",
                            "PREMIUM": st.column_config.NumberColumn("Premi (Rp)", format="Rp %d"),
                            "RATE_PCT": st.column_config.NumberColumn("Rate Aktual", format="%.2f%%"),
                            "mean": st.column_config.NumberColumn("Rata-rata Normal", format="%.2f%%"),
                            "Z_SCORE": st.column_config.NumberColumn("Skor Deviasi", format="%.1f")
                        }
                    )

        with t5:
            c_op1, c_op2 = st.columns([1, 2])
            with c_op1:
                st.subheader("KEPATUHAN SLA")
                fig_sla = px.pie(df['STATUS_SLA'].value_counts().reset_index(), values='count', names='STATUS_SLA', hole=0.5, color_discrete_map={'ON TRACK':C_SUCC, 'DELAYED':C_DANG})
                st.plotly_chart(make_chart(fig_sla), use_container_width=True)
                
            with c_op2:
                st.subheader("BEBAN KERJA INPUT (KINERJA & SLA)")
                perf = df.groupby('INPUT_NAME').agg(
                    PREMIUM=('PREMIUM', 'sum'), 
                    SLA_HARI=('SLA_HARI', 'mean'), 
                    POLICYNO=('POLICYNO', 'count')
                ).reset_index().sort_values('POLICYNO', ascending=False)
                
                top_10_admin = perf.head(10)
                fig_bar_admin = px.bar(
                    top_10_admin, 
                    x='POLICYNO', 
                    y='INPUT_NAME', 
                    orientation='h',
                    text_auto=True,
                    color='SLA_HARI', 
                    color_continuous_scale='RdYlGn_r', 
                    labels={'POLICYNO': 'Jumlah Transaksi', 'SLA_HARI': 'Rata-rata SLA (Hari)'}
                )
                fig_bar_admin.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(make_chart(fig_bar_admin), use_container_width=True)
                
                with st.expander("Klik di sini untuk melihat Rincian SEMUA INPUT"):
                    st.dataframe(
                        perf,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "INPUT_NAME": "Nama Inputf",
                            "POLICYNO": st.column_config.NumberColumn("Total Transaksi", format="%d Polis"),
                            "SLA_HARI": st.column_config.NumberColumn("Rata-rata Kecepatan", format="%.1f Hari"),
                            "PREMIUM": st.column_config.NumberColumn("Total Omset Dipegang (Rp)", format="Rp %d")
                        }
                    )

        with t6:
            st.dataframe(df, use_container_width=True)
            st.download_button("DOWNLOAD DATA CSV", data=df.to_csv(index=False).encode('utf-8'), file_name="export.csv", mime="text/csv")
            
        with t7:
            st.subheader("KINERJA TOP AGEN & BROKER")
            st.caption("KLIK pada salah satu batang grafik (Agen) di bawah untuk melihat rincian transaksinya.")
            
            c_br1, c_br2 = st.columns([2, 1])
            
            df_broker = df.groupby('MO_NAME').agg({'PREMIUM':'sum', 'POLICYNO':'count'}).reset_index()
            df_broker = df_broker[df_broker['PREMIUM'] > 0].sort_values('PREMIUM', ascending=False)
            
            selected_broker = None
            
            with c_br1:
                top_10_broker = df_broker.head(10)
                if not top_10_broker.empty:
                    fig_broker = px.bar(
                        top_10_broker, 
                        x='PREMIUM', 
                        y='MO_NAME', 
                        orientation='h',
                        text_auto='.2s',
                        color='PREMIUM',
                        color_continuous_scale='Blues'
                    )
                    fig_broker.update_traces(hovertemplate='<b>%{y}</b><br>Omset: Rp %{x:,.0f}')
                    fig_broker.update_layout(yaxis={'categoryorder':'total ascending'})
                    
                    event = st.plotly_chart(make_chart(fig_broker), use_container_width=True, on_select="rerun", selection_mode="points")
                    
                    if len(event.selection.points) > 0:
                        selected_broker = event.selection.points[0].y
                else:
                    st.info("Tidak ada data Agen/Broker untuk ditampilkan.")
                    
            with c_br2:
                if selected_broker:
                    st.success(f"**FILTER AKTIF:** {selected_broker}")
                    st.write("Daftar Transaksi Agen:")
                    detail_df = df[df['MO_NAME'] == selected_broker][['INSURED_NAME', 'TOC_DESCRIPTION', 'PREMIUM']]
                    st.dataframe(detail_df, use_container_width=True, hide_index=True)
                else:
                    st.write("**TABEL LENGKAP KONTRIBUTOR:**")
                    st.dataframe(
                        df_broker, 
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "MO_NAME": "Nama Agen / Broker",
                            "PREMIUM": st.column_config.NumberColumn("Total Premi (Rp)", format="Rp %d"),
                            "POLICYNO": "Jml Transaksi"
                        }
                    )

    else:
        st.warning("Data Kosong. Cek filter rentang waktu atau format tanggal.")
else:
    st.info("Silakan masukkan file CSV/Parquet ke folder 'data_produksi'.")

