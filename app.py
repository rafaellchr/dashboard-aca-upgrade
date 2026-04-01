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

# CSS TAMPILAN - TEMA ELEGAN
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 3rem; font-family: 'Segoe UI', Arial, sans-serif;}
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border-left: 6px solid #0F172A;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    div[data-testid="metric-container"] label {
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        color: #0F172A !important;
    }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 700; color: #475569; padding-top: 15px; padding-bottom: 15px;}
    .stTabs [aria-selected="true"] { color: #0F172A !important; border-bottom: 3px solid #0F172A !important;}
</style>
""", unsafe_allow_html=True)

# WARNA TEMA ELEGAN
C_PRIM = "#0F172A" # Dark Navy
C_SEC  = "#3B82F6" # Biru Terang
C_DANG = "#E11D48" # Merah
C_SUCC = "#10B981" # Hijau
C_WARN = "#F59E0B" # Emas/Kuning

# UKURAN FONT GRAFIK DIPERBESAR
def make_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=30, l=10, r=10),
        font=dict(family="Segoe UI, Arial", size=13, color="#334155"),
        xaxis=dict(showgrid=False, linecolor="#cbd5e1", title=""),
        yaxis=dict(showgrid=False, zeroline=False, title=""),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
    )
    return fig

# --- 2. SISTEM LOGIN ---
check_password() 

# --- 3. UI DASHBOARD UTAMA ---
col_logo, col_logout = st.sidebar.columns([5, 3]) 
if col_logout.button("Keluar", use_container_width=True):
    st.session_state['logged_in'] = False
    st.rerun()

st.sidebar.header("PENGATURAN DATA")
date_option = st.sidebar.radio("Format Tanggal:", ["Otomatis (Sistem)", "INDO (Hari/Bulan)", "US (Bulan/Hari)"], index=0)
uploaded_files = st.sidebar.file_uploader("Unggah File Tambahan (CSV/Excel)", type=['csv', 'parquet'], accept_multiple_files=True)

df_raw, error_msg = load_data_hybrid(uploaded_files, date_option)

if df_raw is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("PILIHAN FILTER")
    
    min_date = df_raw['TGL_IN'].min().date()
    max_date = df_raw['TGL_IN'].max().date()
    
    date_range = st.sidebar.date_input("Pilih Rentang Waktu:", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df_raw[(df_raw['TGL_IN'].dt.date >= start_date) & (df_raw['TGL_IN'].dt.date <= end_date)]
    else:
        df = df_raw.copy()
        start_date, end_date = min_date, max_date
    
    all_segments = ["SEMUA SEGMEN"] + sorted(df['SEGMENT'].unique().tolist())
    sel_segment = st.sidebar.selectbox("Pilih Segmen Bisnis:", all_segments)
    if sel_segment != "SEMUA SEGMEN": df = df[df['SEGMENT'] == sel_segment]
    
    all_products = ["SEMUA PRODUK"] + sorted(df['TOC_DESCRIPTION'].unique().tolist())
    sel_product = st.sidebar.selectbox("Pilih Produk:", all_products)
    if sel_product != "SEMUA PRODUK": df = df[df['TOC_DESCRIPTION'] == sel_product]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("SIMULASI TARGET")
    target_pct = st.sidebar.slider("Target Pertumbuhan Tahunan (%)", min_value=5, max_value=100, value=15, step=1)
    
    # --- UI BODY ---
    st.title("DASHBOARD EKSEKUTIF ACA BOGOR")
    st.caption("Ringkasan kinerja bisnis, pemantauan pencapaian target, dan rekomendasi strategis otomatis dari sistem.")
    
    if len(date_range) == 2:
        st.write(f"Periode Pemantauan: {start_date.strftime('%d %B %Y')} s/d {end_date.strftime('%d %B %Y')} | Segmen: {sel_segment} | Produk: {sel_product}")
    
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
        k1.metric(f"PENDAPATAN ({current_year})", f"{curr_omset/1e9:.2f} Miliar", f"{growth_yoy:.1f}% vs Tahun Lalu")
        k2.metric("PENCAPAIAN TARGET", f"{achievment:.1f}%", f"Berdasarkan Target +{target_pct}%")
        k3.metric("RATA-RATA PROSES", f"{df['SLA_HARI'].mean():.1f} Hari", "Target Maks. 2 Hari", delta_color="inverse")
        k4.metric("JUMLAH TRANSAKSI", f"{len(df):,} Polis", "Aktivitas Saat Ini")

        st.write("Silakan tekan tombol di bawah untuk mengunduh laporan ringkasan manajerial secara lengkap.")
        
        top_prod_report = df.groupby('TOC_DESCRIPTION')['PREMIUM'].sum().nlargest(5).reset_index()
        prod_list_html = "".join([f"<tr><td style='padding:8px; border-bottom:1px solid #ddd;'>{row['TOC_DESCRIPTION']}</td><td style='padding:8px; border-bottom:1px solid #ddd; text-align:right;'>Rp {row['PREMIUM']/1e6:,.0f} Juta</td></tr>" for _, row in top_prod_report.iterrows()])
        
        top_broker_report = df.groupby('MO_NAME')['PREMIUM'].sum().nlargest(5).reset_index()
        broker_list_html = "".join([f"<tr><td style='padding:8px; border-bottom:1px solid #ddd;'>{row['MO_NAME']}</td><td style='padding:8px; border-bottom:1px solid #ddd; text-align:right;'>Rp {row['PREMIUM']/1e6:,.0f} Juta</td></tr>" for _, row in top_broker_report.iterrows()])
        
        avg_sla = df['SLA_HARI'].mean()
        
        report_html = f"""
        <html><head><title>Laporan Eksekutif ACA Bogor</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; padding: 40px; color: #334155; line-height: 1.6; }}
            h1 {{ color: #0F172A; border-bottom: 3px solid #3b82f6; padding-bottom: 10px; font-size: 24px; }}
            h2 {{ color: #0F172A; margin-top: 30px; border-bottom: 1px solid #cbd5e1; padding-bottom: 5px; font-size: 18px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }}
            th {{ background-color: #f1f5f9; text-align: left; padding: 10px; font-weight: bold; color: #475569; }}
            .highlight {{ background-color: #f8fafc; padding: 15px; border-left: 5px solid #0F172A; border-radius: 5px; margin-bottom: 20px; }}
            .metric-container {{ display: flex; justify-content: space-between; margin-top: 15px; }}
            .metric-box {{ width: 31%; background: #f8fafc; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e2e8f0; }}
            .metric-title {{ font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: bold; }}
            .metric-value {{ font-size: 22px; font-weight: bold; color: #0f172a; margin-top: 5px; }}
        </style>
        </head>
        <body>
            <h1>LAPORAN EKSEKUTIF KINERJA ACA BOGOR</h1>
            <p><b>Dicetak pada:</b> {datetime.now().strftime('%d %B %Y, %H:%M WIB')}</p>
            <p><b>Periode Data:</b> {start_date.strftime('%d %b %Y')} s/d {end_date.strftime('%d %b %Y')}</p>
            <p><b>Segmen Terpilih:</b> {sel_segment} | <b>Produk Terpilih:</b> {sel_product}</p>
            
            <div class="highlight">
                <p style="margin:0;">Laporan ini merupakan ringkasan manajerial dari performa bisnis, mencakup analisis pendapatan, efisiensi operasional, dan kontributor utama untuk mempermudah pengambilan keputusan.</p>
            </div>

            <h2>1. RINGKASAN PERFORMA BISNIS</h2>
            <div class="metric-container">
                <div class="metric-box">
                    <div class="metric-title">Total Pendapatan</div>
                    <div class="metric-value">Rp {curr_omset:,.0f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Pertumbuhan (YoY)</div>
                    <div class="metric-value">{growth_yoy:.2f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Total Transaksi</div>
                    <div class="metric-value">{len(df):,} Polis</div>
                </div>
            </div>

            <h2>2. EFISIENSI OPERASIONAL & TARGET</h2>
            <ul>
                <li><b>Rata-rata Durasi Proses:</b> {avg_sla:.1f} Hari kerja per transaksi.</li>
                <li><b>Status Kinerja:</b> {'Memenuhi Target' if avg_sla < 2 else 'Perlu Perhatian (Melebihi Target 2 Hari)'}</li>
                <li><b>Pencapaian Target:</b> {achievment:.1f}% (Berdasarkan asumsi target tahunan {target_pct}%)</li>
            </ul>

            <h2>3. LIMA PRODUK PENYUMBANG TERBESAR</h2>
            <table>
                <tr><th>Nama Produk</th><th style="text-align:right;">Total Pendapatan</th></tr>
                {prod_list_html}
            </table>

            <h2>4. LIMA AGEN / BROKER KONTRIBUTOR TERBESAR</h2>
            <table>
                <tr><th>Nama Agen / Broker</th><th style="text-align:right;">Total Pendapatan</th></tr>
                {broker_list_html}
            </table>
            
            <br><br>
            <p style="text-align:center; font-size:12px; color:#94a3b8; margin-top:40px;">Dokumen dicetak secara otomatis melalui Sistem Eksekutif ACA Bogor</p>
        </body></html>
        """
        st.download_button(label="Unduh Laporan Eksekutif", data=report_html, file_name=f"Laporan_Eksekutif_ACABogor_{datetime.now().strftime('%Y%m%d')}.html", mime="text/html")

        st.markdown("---")
        
        # PERUBAHAN: Urutan tab diubah dan penamaannya disederhanakan
        t1, t2, t3, t4, t5, t6, t7 = st.tabs([
            "TREN PENDAPATAN", 
            "PELUANG & RISIKO (AI)", 
            "PRODUK & PORTOFOLIO", 
            "ANALISIS NASABAH", 
            "KINERJA OPERASIONAL", 
            "MITRA AGEN", 
            "DATA MENTAH"
        ])
        
        with t1:
            c1, c2 = st.columns([2,1])
            with c1:
                st.subheader("Pergerakan Pendapatan")
                trend = df.groupby(['TAHUN', 'BULAN_NUM', 'BULAN_NAMA'])['PREMIUM'].sum().reset_index().sort_values(['TAHUN', 'BULAN_NUM'])
                fig = px.line(trend, x='BULAN_NAMA', y='PREMIUM', color='TAHUN', markers=True, color_discrete_sequence=[C_SEC, C_PRIM])
                fig.update_traces(line=dict(width=3), marker=dict(size=8))
                st.plotly_chart(make_chart(fig), use_container_width=True)
            with c2:
                st.subheader("Rata-rata Pola Bulanan")
                seas = df.groupby('BULAN_NAMA')['PREMIUM'].mean().reindex(['JANUARY','FEBRUARY','MARCH','APRIL','MAY','JUNE','JULY','AUGUST','SEPTEMBER','OCTOBER','NOVEMBER','DECEMBER']).reset_index()
                fig_s = px.bar(seas, x='BULAN_NAMA', y='PREMIUM', color='PREMIUM', color_continuous_scale='Blues')
                fig_s.update_layout(coloraxis_showscale=False)
                st.plotly_chart(make_chart(fig_s), use_container_width=True)
            
            st.markdown("---")
            st.subheader("Prediksi Masa Depan (Sistem Otomatis)")
            st.caption("Memprediksi estimasi pendapatan hingga 6 bulan ke depan berdasarkan pola data historis.")
            
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
                        forecast_type = 'Prediksi Sistem'
                    except:
                        z = np.polyfit(np.arange(len(df_ts)), df_ts['PREMIUM'].values, 1)
                        pred_vals = [np.poly1d(z)(len(df_ts) - 1 + i) for i in range(1, future_steps + 1)]
                        forecast_type = 'Prediksi Sistem (Linear)'
                else:
                    z = np.polyfit(np.arange(len(df_ts)), df_ts['PREMIUM'].values, 1)
                    pred_vals = [np.poly1d(z)(len(df_ts) - 1 + i) for i in range(1, future_steps + 1)]
                    forecast_type = 'Prediksi Sistem (Linear)'

                for i, p_val in enumerate(pred_vals):
                    next_month = last_month + (i + 1)
                    next_year = last_year + (next_month - 1) // 12
                    next_month = (next_month - 1) % 12 + 1
                    pred_list.append({'PERIODE': f"{next_year}-{str(next_month).zfill(2)}", 'PREMIUM': max(p_val, 0), 'TIPE': forecast_type})
                
                df_ts['TIPE'] = 'Data Saat Ini'
                df_combined = pd.concat([df_ts[['PERIODE', 'PREMIUM', 'TIPE']].tail(12), pd.DataFrame(pred_list)])
                fig_cast = px.bar(df_combined, x='PERIODE', y='PREMIUM', color='TIPE', color_discrete_map={'Data Saat Ini': C_PRIM, forecast_type: C_WARN})
                st.plotly_chart(make_chart(fig_cast), use_container_width=True)
            else:
                st.info("Pilih rentang waktu minimal 6 bulan untuk mengaktifkan fitur Prediksi.")

        with t2:
            st.subheader("Temuan Sistem Otomatis")
            st.info("Menu ini dirancang secara khusus untuk melakukan pemindaian otomatis pada seluruh data Anda guna memetakan peluang penjualan baru dan mendeteksi anomali pada harga.")
            st.markdown("---")
            
            st.write("**1. Peluang Penjualan Produk Tambahan**")
            st.caption("Daftar pasangan produk yang paling sering dibeli secara bersamaan oleh nasabah yang sama. Tawarkan paket ini ke nasabah lain untuk meningkatkan pendapatan.")
            basket = df.groupby(['INSURED_NAME', 'TOC_DESCRIPTION'])['POLICYNO'].count().unstack().fillna(0).map(lambda x: 1 if x > 0 else 0)
            
            if len(basket.columns) > 1:
                co_matrix = basket.T.dot(basket)
                
                # Memastikan tidak terjadi error read-only array
                arr = co_matrix.to_numpy(copy=True)
                np.fill_diagonal(arr, 0)
                co_matrix = pd.DataFrame(arr, index=co_matrix.index, columns=co_matrix.columns)
                
                stacked = co_matrix.stack()
                stacked.index.names = ['Product A', 'Product B'] 
                pairs = stacked.reset_index(name='Count')
                pairs = pairs[pairs['Count'] > 0]
                pairs['sorted_key'] = pairs.apply(lambda x: tuple(sorted([x['Product A'], x['Product B']])), axis=1)
                top_pairs = pairs.drop_duplicates(subset='sorted_key').sort_values('Count', ascending=False).head(10)
                
                if not top_pairs.empty:
                    top_pairs['Full Name'] = top_pairs['Product A'] + " + " + top_pairs['Product B']
                    top_pairs['Pair Name'] = top_pairs['Product A'].str.slice(0, 22) + "... + " + top_pairs['Product B'].str.slice(0, 22) + "..."
                    
                    fig_pairs = px.bar(
                        top_pairs, 
                        x='Count', 
                        y='Pair Name', 
                        orientation='h',
                        hover_name='Full Name',
                        color_discrete_sequence=[C_SEC]
                    )
                    fig_pairs.update_layout(yaxis={'title': ''}, xaxis={'title': 'Jumlah Pembelian Bersamaan'})
                    st.plotly_chart(make_chart(fig_pairs), use_container_width=True)
                else: 
                    st.warning("Belum ada data nasabah yang membeli dua produk berbeda.")
            else: 
                st.warning("Variasi produk belum cukup untuk analisis ini.")

            st.markdown("---")
            
            st.write("**2. Peringatan Harga Tidak Wajar (Deteksi Anomali)**")
            st.caption("Titik berwarna Merah mendeteksi transaksi dengan Rate (%) yang terlalu tinggi atau rendah dibandingkan rata-rata normal. Tolong periksa ulang untuk mencegah kesalahan ketik.")
            
            stats = df[df['RATE_PCT'] > 0].groupby('TOC_DESCRIPTION')['RATE_PCT'].agg(['mean', 'std']).reset_index()
            df_risk = pd.merge(df, stats, on='TOC_DESCRIPTION', how='left')
            df_risk['Z_SCORE'] = (df_risk['RATE_PCT'] - df_risk['mean']) / df_risk['std']
            
            df_risk['IS_ANOMALY'] = np.where((df_risk['Z_SCORE'].abs() > 3) & (df_risk['PREMIUM'] > 1000000), 'Perlu Diperiksa', 'Normal')
            
            c_anom1, c_anom2 = st.columns([2, 1])
            
            with c_anom1:
                df_plot_anom = df_risk[df_risk['PREMIUM'] > 0]
                
                fig_anom = px.scatter(
                    df_plot_anom, 
                    x="PREMIUM", 
                    y="RATE_PCT", 
                    color="IS_ANOMALY", 
                    color_discrete_map={'Perlu Diperiksa': C_DANG, 'Normal': '#e2e8f0'}, 
                    hover_name="INSURED_NAME",
                    hover_data={
                        "POLICYNO": True, 
                        "TOC_DESCRIPTION": True, 
                        "RATE_PCT": ":.2f", 
                        "IS_ANOMALY": False,
                        "PREMIUM": ":,.0f"
                    },
                    labels={'PREMIUM': 'Nilai Transaksi (Rp)', 'RATE_PCT': 'Rate (%)'},
                    log_x=True 
                )
                
                fig_anom.update_layout(xaxis=dict(tickformat='.0s', title="Besar Transaksi"), yaxis=dict(title="Rate (%)"))
                fig_anom.update_traces(marker=dict(size=8, line=dict(width=0))) 
                st.plotly_chart(make_chart(fig_anom), use_container_width=True)
                
            with c_anom2:
                anomalies_df = df_risk[df_risk['IS_ANOMALY'] == 'Perlu Diperiksa'].sort_values('Z_SCORE', key=abs, ascending=False)
                st.write(f"**Peringatan: {len(anomalies_df)} Transaksi Terdeteksi Tidak Wajar**")
                
                if not anomalies_df.empty:
                    st.dataframe(
                        anomalies_df[['POLICYNO', 'RATE_PCT']],
                        use_container_width=True,
                        hide_index=True,
                        column_config={"RATE_PCT": st.column_config.NumberColumn("Rate (%)", format="%.2f")}
                    )
                else:
                    st.success("Aman. Tidak ada indikasi anomali pada rate premi saat ini.")
                    
            if not anomalies_df.empty:
                with st.expander("Buka Rincian Lengkap Data yang Perlu Diperiksa"):
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

        with t3:
            st.subheader("Analisis Dominasi Produk")
            c_p1, c_p2 = st.columns([1, 1]) 
            
            with c_p1:
                # PERUBAHAN: Treemap diganti dengan Pie/Donut Chart agar jauh lebih bersih dan mudah dibaca
                st.write("**PROPORSI BERDASARKAN SEGMEN BISNIS**")
                df_seg = df.groupby('SEGMENT')['PREMIUM'].sum().reset_index()
                if not df_seg.empty:
                    fig_pie_seg = px.pie(df_seg, values='PREMIUM', names='SEGMENT', hole=0.4, color_discrete_sequence=px.colors.qualitative.Prism)
                    fig_pie_seg.update_traces(textinfo='percent+label', textfont_size=14)
                    st.plotly_chart(make_chart(fig_pie_seg), use_container_width=True)
            
            with c_p2:
                st.write("**SEPULUH PRODUK PENYUMBANG TERBESAR**")
                
                list_segmen = ["SEMUA SEGMEN"] + list(df['SEGMENT'].dropna().unique())
                pilihan_segmen = st.selectbox("Saring berdasarkan Segmen:", list_segmen)
                
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
                    
                    fig_bar = make_chart(fig_bar)
                    
                    fig_bar.update_traces(
                        texttemplate='Rp %{text:,.0f}', 
                        textposition='inside', 
                        insidetextanchor='middle',
                        textfont=dict(color='white', size=13)
                    )
                    fig_bar.update_layout(
                        yaxis={'categoryorder': 'total ascending', 'title': ''},
                        xaxis={'visible': False}, 
                        height=550 
                    )
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                
        with t4:
            st.subheader("Analisis Status Nasabah")
            snap_date = df['TGL_IN'].max() + timedelta(days=1)
            rfm = df.groupby('INSURED_NAME').agg({'TGL_IN': lambda x: (snap_date - x.max()).days, 'POLICYNO': 'count', 'PREMIUM': 'sum'}).rename(columns={'TGL_IN': 'RECENCY', 'POLICYNO': 'FREQ', 'PREMIUM': 'MONETARY'})
            
            # Label disederhanakan
            rfm['STATUS'] = rfm.apply(lambda r: "PELANGGAN VIP" if r['RECENCY'] < 180 and r['MONETARY'] > 100000000 else ("AKTIF" if r['RECENCY'] < 180 else ("BERISIKO PINDAH" if r['RECENCY'] < 365 else "TIDAK AKTIF")), axis=1)
            
            rfm['IS_CHURN_HIST'] = (rfm['RECENCY'] > 180).astype(int)
            if len(rfm['IS_CHURN_HIST'].unique()) > 1:
                X = rfm[['FREQ', 'MONETARY']]
                y = rfm['IS_CHURN_HIST']
                rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                rf_model.fit(X, y)
                rfm['PROB_CHURN_%'] = rf_model.predict_proba(X)[:, 1] * 100
            else:
                rfm['PROB_CHURN_%'] = 0.0

            c_rfm1, c_rfm2 = st.columns([1,2])
            with c_rfm1:
                st.write("**Komposisi Status Nasabah Saat Ini**")
                fig_pie = px.pie(rfm['STATUS'].value_counts().reset_index(), values='count', names='STATUS', hole=0.5, 
                                 color='STATUS', color_discrete_map={'PELANGGAN VIP': C_SUCC, 'AKTIF': C_SEC, 'BERISIKO PINDAH': C_WARN, 'TIDAK AKTIF': C_DANG})
                st.plotly_chart(make_chart(fig_pie), use_container_width=True)
            with c_rfm2:
                st.write("**Peringatan Dini: Nasabah Aktif Berisiko Pindah**")
                st.caption("Sistem mendeteksi nasabah di bawah ini berpotensi berhenti memperpanjang asuransi berdasarkan riwayat mereka.")
                
                at_risk = rfm[rfm['STATUS'] == 'AKTIF'].sort_values('PROB_CHURN_%', ascending=False).head(10).reset_index()
                
                if not at_risk.empty:
                    st.dataframe(
                        at_risk[['INSURED_NAME', 'PROB_CHURN_%', 'FREQ', 'MONETARY']], 
                        use_container_width=True,
                        hide_index=True, 
                        column_config={
                            "INSURED_NAME": st.column_config.TextColumn("Nama Nasabah", width="medium"), 
                            "PROB_CHURN_%": st.column_config.ProgressColumn("Risiko Pindah", min_value=0, max_value=100, format="%.1f%%"),
                            "FREQ": st.column_config.NumberColumn("Jumlah Transaksi"),
                            "MONETARY": st.column_config.NumberColumn("Total Pembelian (Rp)", format="Rp %d")
                        }
                    )
                else:
                    st.success("Belum ada indikasi nasabah aktif yang berisiko tinggi untuk pindah saat ini.")
            
            st.markdown("---")
            st.write("**Peta Evaluasi Nasabah**")
            st.caption("Petunjuk: Semakin posisinya berada di bawah (jarak transaksi dekat) dan di kanan (sering bertransaksi), maka nasabah tersebut semakin setia dan berharga.")
            
            rfm_plot = rfm.reset_index().copy()
            rfm_plot['MONETARY_SIZE'] = rfm_plot['MONETARY'].abs() + 1 
            
            c_fil1, c_fil2 = st.columns([1, 4])
            
            with c_fil1:
                st.caption("Saring Grafik:")
                filter_status = st.radio(
                    "Pilih Status:", 
                    ["TAMPILKAN SEMUA", "PELANGGAN VIP", "AKTIF", "BERISIKO PINDAH", "TIDAK AKTIF"]
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
                            'PELANGGAN VIP': C_SUCC, 
                            'AKTIF': C_SEC, 
                            'BERISIKO PINDAH': C_WARN, 
                            'TIDAK AKTIF': C_DANG
                        },
                        labels={'RECENCY': 'Jarak Sejak Transaksi Terakhir (Hari)', 'FREQ': 'Total Frekuensi Transaksi'}
                    )
                    st.plotly_chart(make_chart(fig_2d), use_container_width=True)

        with t5:
            st.subheader("Evaluasi Kinerja Operasional")
            c_op1, c_op2 = st.columns([1, 2])
            with c_op1:
                st.write("**Kepatuhan Durasi Proses Dokumen**")
                fig_sla = px.pie(df['STATUS_SLA'].value_counts().reset_index(), values='count', names='STATUS_SLA', hole=0.5, color='STATUS_SLA', color_discrete_map={'ON TRACK':C_SUCC, 'DELAYED':C_DANG, 'AMAN': C_SUCC, 'TERLAMBAT': C_DANG})
                st.plotly_chart(make_chart(fig_sla), use_container_width=True)
                
            with c_op2:
                st.write("**Evaluasi Kecepatan Staf Administrasi**")
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
                    labels={'POLICYNO': 'Jumlah Dokumen Diproses', 'SLA_HARI': 'Rata-rata Lama Proses (Hari)'}
                )
                fig_bar_admin.update_layout(yaxis={'categoryorder':'total ascending', 'title': ''}, xaxis={'title': ''})
                st.plotly_chart(make_chart(fig_bar_admin), use_container_width=True)
                
                with st.expander("Klik di sini untuk melihat Rincian Lengkap Seluruh Staf"):
                    st.dataframe(
                        perf,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "INPUT_NAME": "Nama Staf Admin",
                            "POLICYNO": st.column_config.NumberColumn("Total Pekerjaan", format="%d Dokumen"),
                            "SLA_HARI": st.column_config.NumberColumn("Rata-rata Durasi", format="%.1f Hari"),
                            "PREMIUM": st.column_config.NumberColumn("Nilai Transaksi Dikelola (Rp)", format="Rp %d")
                        }
                    )
            
        with t6:
            st.subheader("Peringkat Mitra Agen dan Broker")
            st.caption("Silakan klik salah satu batang grafik di bawah untuk melihat rincian riwayat transaksi agen tersebut.")
            
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
                        color='PREMIUM',
                        color_continuous_scale='Blues'
                    )
                    fig_broker.update_traces(
                        texttemplate='Rp %{x:,.0f}',
                        textposition='inside',
                        insidetextanchor='middle',
                        textfont=dict(color='white', size=13),
                        hovertemplate='<b>%{y}</b><br>Omset: Rp %{x:,.0f}'
                    )
                    fig_broker.update_layout(
                        yaxis={'categoryorder':'total ascending', 'title': ''}, 
                        xaxis={'visible': False}
                    )
                    
                    event = st.plotly_chart(make_chart(fig_broker), use_container_width=True, on_select="rerun", selection_mode="points")
                    
                    if len(event.selection.points) > 0:
                        selected_broker = event.selection.points[0].y
                else:
                    st.info("Tidak ada data Mitra Agen untuk ditampilkan.")
                    
            with c_br2:
                if selected_broker:
                    st.success(f"Filter Diterapkan: {selected_broker}")
                    st.write("Daftar Transaksi Baru-baru Ini:")
                    detail_df = df[df['MO_NAME'] == selected_broker][['INSURED_NAME', 'TOC_DESCRIPTION', 'PREMIUM']]
                    st.dataframe(detail_df, use_container_width=True, hide_index=True)
                else:
                    st.write("**Tabel Rekapitulasi Kontributor:**")
                    st.dataframe(
                        df_broker, 
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "MO_NAME": "Nama Agen / Broker",
                            "PREMIUM": st.column_config.NumberColumn("Total Pendapatan (Rp)", format="Rp %d"),
                            "POLICYNO": "Jumlah Transaksi"
                        }
                    )

        with t7:
            st.subheader("Data Transaksi Mentah")
            st.caption("Tabel ini berisi seluruh data mentah jika Anda membutuhkan pengecekan manual yang spesifik.")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Unduh Data ke Format Excel (CSV)", data=df.to_csv(index=False).encode('utf-8'), file_name="Data_Transaksi_ACABogor.csv", mime="text/csv")

    else:
        st.warning("Data belum tersedia. Silakan periksa kembali filter rentang waktu yang Anda pilih atau format tanggal sistem.")
else:
    st.info("Silakan unggah dokumen data pada panel sebelah kiri atau masukkan file ke folder data_produksi.")
