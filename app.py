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

# CSS TAMPILAN - TEMA ELEGAN (NAVY & GOLD)
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; font-family: 'Segoe UI', sans-serif;}
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border-left: 6px solid #0F172A; /* Navy Blue Elegant */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    div[data-testid="metric-container"] label {
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        color: #0F172A !important;
    }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; color: #475569; padding-top: 15px; padding-bottom: 15px;}
    .stTabs [aria-selected="true"] { color: #0F172A !important; border-bottom: 3px solid #0F172A !important;}
    h1, h2, h3 { color: #0F172A; }
</style>
""", unsafe_allow_html=True)

# WARNA TEMA BARU YANG LEBIH MATANG
C_PRIM = "#0F172A" # Dark Navy (Utama)
C_SEC  = "#3B82F6" # Bright Blue (Sekunder)
C_DANG = "#E11D48" # Merah elegan (Bahaya/Terlambat)
C_SUCC = "#10B981" # Hijau elegan (Aman/Bagus)
C_WARN = "#F59E0B" # Emas/Kuning elegan (Peringatan/Peluang)

# FUNGSI CHART DIPERBESAR FONTNYA
def make_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=30, l=10, r=10),
        font=dict(family="Segoe UI, Arial", size=13, color="#334155"), # Ukuran font dibesarkan
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
uploaded_files = st.sidebar.file_uploader("Unggah Data Transaksi (CSV/Excel)", type=['csv', 'parquet'], accept_multiple_files=True)

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
    st.sidebar.subheader("SIMULASI TARGET TAHUNAN")
    target_pct = st.sidebar.slider("Target Pertumbuhan (%)", min_value=5, max_value=100, value=15, step=1)
    
    st.sidebar.success(f"✔️ {len(df):,} Transaksi berhasil dimuat.")
    
    # --- UI BODY ---
    st.title("DASHBOARD EKSEKUTIF ACA BOGOR")
    st.caption("Ringkasan kinerja bisnis, pemantauan pencapaian target, dan rekomendasi strategis berbasis Kecerdasan Buatan (AI).")
    
    if len(date_range) == 2:
        st.write(f"**Periode Data:** {start_date.strftime('%d %B %Y')} s/d {end_date.strftime('%d %B %Y')} | **Segmen:** {sel_segment}")
    
    if len(df) > 0:
        current_year = df['TGL_IN'].dt.year.max()
        curr_ytd_df = df[df['TGL_IN'].dt.year == current_year]
        prev_year_df = df_raw[(df_raw['TGL_IN'].dt.year == current_year - 1) & (df_raw['TGL_IN'].dt.month <= df['TGL_IN'].dt.month.max())]
        
        curr_omset = curr_ytd_df['PREMIUM'].sum()
        prev_omset = prev_year_df['PREMIUM'].sum()
        growth_yoy = ((curr_omset - prev_omset)/prev_omset * 100) if prev_omset > 0 else 0
        target_omset = prev_omset * (1 + (target_pct / 100))
        achievment = (curr_omset / target_omset * 100) if target_omset > 0 else 0
        
        # PERUBAHAN TATA LETAK METRIK & BAHASA
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"PENDAPATAN ({current_year})", f"{curr_omset/1e9:.2f} M", f"{growth_yoy:.1f}% vs Tahun Lalu")
        k2.metric("PENCAPAIAN TARGET", f"{achievment:.1f}%", f"Dari target naik +{target_pct}%")
        k3.metric("RATA-RATA PROSES", f"{df['SLA_HARI'].mean():.1f} HARI", "Target Maks. 2 Hari", delta_color="inverse")
        k4.metric("JUMLAH TRANSAKSI", f"{len(df):,} Polis", "Aktivitas Bisnis")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # PERUBAHAN: Tab disusun ulang, AI dipindah ke urutan 2 agar mudah dibaca eksekutif
        t1, t2, t3, t4, t5, t6 = st.tabs([
            "📈 TREN & PREDIKSI", 
            "💡 PELUANG & RISIKO (AI)", 
            "📦 PRODUK & SEGMEN", 
            "👥 ANALISIS NASABAH", 
            "⚙️ OPERASIONAL & MITRA", 
            "📂 DATA MENTAH"
        ])
        
        with t1:
            st.subheader("Tren Pendapatan Bulanan")
            st.caption("Grafik ini menunjukkan pergerakan pendapatan kita dari bulan ke bulan.")
            c1, c2 = st.columns([2,1])
            with c1:
                trend = df.groupby(['TAHUN', 'BULAN_NUM', 'BULAN_NAMA'])['PREMIUM'].sum().reset_index().sort_values(['TAHUN', 'BULAN_NUM'])
                fig = px.line(trend, x='BULAN_NAMA', y='PREMIUM', color='TAHUN', markers=True, color_discrete_sequence=[C_SEC, C_PRIM])
                fig.update_traces(line=dict(width=3), marker=dict(size=8))
                st.plotly_chart(make_chart(fig), use_container_width=True)
            with c2:
                seas = df.groupby('BULAN_NAMA')['PREMIUM'].mean().reindex(['JANUARY','FEBRUARY','MARCH','APRIL','MAY','JUNE','JULY','AUGUST','SEPTEMBER','OCTOBER','NOVEMBER','DECEMBER']).reset_index()
                fig_s = px.bar(seas, x='BULAN_NAMA', y='PREMIUM', color='PREMIUM', color_continuous_scale='Blues')
                fig_s.update_layout(coloraxis_showscale=False)
                st.plotly_chart(make_chart(fig_s), use_container_width=True)
            
            st.markdown("---")
            st.subheader("🔮 Prediksi Masa Depan (Oleh AI)")
            st.info("Berdasarkan pola data di atas, sistem memprediksi perkiraan pendapatan kita hingga 6 bulan ke depan. Batang berwarna **Emas** adalah angka prediksinya.")
            
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
                        forecast_type = 'Prediksi Sistem (Garis Lurus)'
                else:
                    z = np.polyfit(np.arange(len(df_ts)), df_ts['PREMIUM'].values, 1)
                    pred_vals = [np.poly1d(z)(len(df_ts) - 1 + i) for i in range(1, future_steps + 1)]
                    forecast_type = 'Prediksi Sistem (Garis Lurus)'

                for i, p_val in enumerate(pred_vals):
                    next_month = last_month + (i + 1)
                    next_year = last_year + (next_month - 1) // 12
                    next_month = (next_month - 1) % 12 + 1
                    pred_list.append({'PERIODE': f"{next_year}-{str(next_month).zfill(2)}", 'PREMIUM': max(p_val, 0), 'TIPE': forecast_type})
                
                df_ts['TIPE'] = 'Data Aktual (Saat Ini)'
                df_combined = pd.concat([df_ts[['PERIODE', 'PREMIUM', 'TIPE']].tail(12), pd.DataFrame(pred_list)])
                fig_cast = px.bar(df_combined, x='PERIODE', y='PREMIUM', color='TIPE', color_discrete_map={'Data Aktual (Saat Ini)': C_PRIM, forecast_type: C_WARN})
                st.plotly_chart(make_chart(fig_cast), use_container_width=True)
            else:
                st.warning("Data masih kurang dari 6 bulan. Prediksi belum bisa dilakukan.")

        # PERUBAHAN: TAB AI DIMAJUKAN & DISIMPLIFIKASI
        with t2:
            st.markdown("### Temuan Sistem Otomatis (AI)")
            st.write("Sistem telah memeriksa seluruh data Anda dan menemukan beberapa pola penting berikut:")
            st.markdown("---")
            
            c_ai1, c_ai2 = st.columns(2)
            
            with c_ai1:
                st.subheader("1. Peluang Jual Produk Tambahan")
                st.caption("Daftar pasangan produk yang paling sering dibeli secara bersamaan oleh nasabah yang sama. Tawarkan paket ini ke nasabah lain!")
                basket = df.groupby(['INSURED_NAME', 'TOC_DESCRIPTION'])['POLICYNO'].count().unstack().fillna(0).map(lambda x: 1 if x > 0 else 0)
                
                if len(basket.columns) > 1:
                    co_matrix = basket.T.dot(basket)
                    arr = co_matrix.to_numpy(copy=True)
                    np.fill_diagonal(arr, 0)
                    co_matrix = pd.DataFrame(arr, index=co_matrix.index, columns=co_matrix.columns)
                    
                    stacked = co_matrix.stack()
                    stacked.index.names = ['Product A', 'Product B'] 
                    pairs = stacked.reset_index(name='Count')
                    pairs = pairs[pairs['Count'] > 0]
                    pairs['sorted_key'] = pairs.apply(lambda x: tuple(sorted([x['Product A'], x['Product B']])), axis=1)
                    top_pairs = pairs.drop_duplicates(subset='sorted_key').sort_values('Count', ascending=False).head(5)
                    
                    if not top_pairs.empty:
                        top_pairs['Full Name'] = top_pairs['Product A'] + " + " + top_pairs['Product B']
                        top_pairs['Pair Name'] = top_pairs['Product A'].str.slice(0, 15) + " + " + top_pairs['Product B'].str.slice(0, 15)
                        
                        fig_pairs = px.bar(top_pairs, x='Count', y='Pair Name', orientation='h', hover_name='Full Name', color_discrete_sequence=[C_SUCC])
                        fig_pairs.update_layout(yaxis={'title': ''}, xaxis={'title': 'Jumlah Transaksi Bersamaan'})
                        st.plotly_chart(make_chart(fig_pairs), use_container_width=True)
                    else: 
                        st.info("Belum ada nasabah yang membeli dua produk berbeda.")
                else: 
                    st.info("Variasi produk belum cukup untuk analisis ini.")

            with c_ai2:
                st.subheader("2. Peringatan Harga Tidak Wajar")
                st.caption("Titik berwarna **Merah** di bawah ini adalah transaksi yang rate/harganya terlalu jauh dari rata-rata normal. Tolong periksa ulang apakah ada diskon berlebih atau salah ketik.")
                
                stats = df[df['RATE_PCT'] > 0].groupby('TOC_DESCRIPTION')['RATE_PCT'].agg(['mean', 'std']).reset_index()
                df_risk = pd.merge(df, stats, on='TOC_DESCRIPTION', how='left')
                df_risk['Z_SCORE'] = (df_risk['RATE_PCT'] - df_risk['mean']) / df_risk['std']
                df_risk['IS_ANOMALY'] = np.where((df_risk['Z_SCORE'].abs() > 3) & (df_risk['PREMIUM'] > 1000000), 'Perlu Diperiksa (Anomali)', 'Normal')
                
                df_plot_anom = df_risk[df_risk['PREMIUM'] > 0]
                
                fig_anom = px.scatter(
                    df_plot_anom, x="PREMIUM", y="RATE_PCT", color="IS_ANOMALY", 
                    color_discrete_map={'Perlu Diperiksa (Anomali)': C_DANG, 'Normal': '#CBD5E1'}, 
                    hover_name="INSURED_NAME",
                    hover_data={"POLICYNO": True, "TOC_DESCRIPTION": True, "RATE_PCT": ":.2f", "IS_ANOMALY": False, "PREMIUM": ":,.0f"},
                    labels={'PREMIUM': 'Nilai Transaksi (Rp)', 'RATE_PCT': 'Rate Harga (%)'},
                    log_x=True 
                )
                fig_anom.update_layout(xaxis=dict(tickformat='.0s', title="Besar Transaksi"), yaxis=dict(title="Rate (%)"))
                fig_anom.update_traces(marker=dict(size=8, line=dict(width=0))) 
                st.plotly_chart(make_chart(fig_anom), use_container_width=True)

        # PERUBAHAN: TAB PRODUK - Treemap diganti Donut Chart agar mudah dibaca
        with t3:
            st.subheader("Dominasi Segmen & Top Produk")
            c_p1, c_p2 = st.columns([1, 1]) 
            
            with c_p1:
                st.write("**PROPORSI BERDASARKAN SEGMEN BISNIS**")
                df_seg = df.groupby('SEGMENT')['PREMIUM'].sum().reset_index()
                if not df_seg.empty:
                    fig_pie_seg = px.pie(df_seg, values='PREMIUM', names='SEGMENT', hole=0.4, color_discrete_sequence=px.colors.qualitative.Prism)
                    fig_pie_seg.update_traces(textinfo='percent+label', textfont_size=14)
                    st.plotly_chart(make_chart(fig_pie_seg), use_container_width=True)
            
            with c_p2:
                st.write("**TOP 10 PRODUK PENYUMBANG PENDAPATAN**")
                list_segmen = ["SEMUA SEGMEN"] + list(df['SEGMENT'].dropna().unique())
                pilihan_segmen = st.selectbox("Saring berdasarkan Segmen:", list_segmen)
                
                if pilihan_segmen == "SEMUA SEGMEN": df_plot_base = df 
                else: df_plot_base = df[df['SEGMENT'] == pilihan_segmen]
                
                if df_plot_base.empty:
                    st.info(f"Belum ada transaksi untuk segmen: {pilihan_segmen}.")
                else:
                    df_grouped = df_plot_base.groupby('TOC_DESCRIPTION')['PREMIUM'].sum().reset_index()
                    df_grouped = df_grouped.sort_values('PREMIUM', ascending=False)
                    
                    top_n = 10
                    if len(df_grouped) > top_n:
                        df_top = df_grouped.iloc[:top_n]
                        df_others = pd.DataFrame({'TOC_DESCRIPTION': ['LAINNYA (GABUNGAN)'], 'PREMIUM': [df_grouped.iloc[top_n:]['PREMIUM'].sum()]})
                        df_plot = pd.concat([df_top, df_others], ignore_index=True)
                    else:
                        df_plot = df_grouped

                    fig_bar = px.bar(df_plot, x='PREMIUM', y='TOC_DESCRIPTION', orientation='h', text='PREMIUM', color_discrete_sequence=[C_PRIM])
                    fig_bar = make_chart(fig_bar)
                    fig_bar.update_traces(texttemplate='Rp %{text:,.0f}', textposition='inside', insidetextanchor='middle', textfont=dict(color='white', size=12))
                    fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, xaxis={'visible': False}, height=500)
                    st.plotly_chart(fig_bar, use_container_width=True)

        # PERUBAHAN: TAB NASABAH - Jargon Disederhanakan
        with t4:
            st.subheader("Analisis Perilaku Nasabah")
            snap_date = df['TGL_IN'].max() + timedelta(days=1)
            rfm = df.groupby('INSURED_NAME').agg({'TGL_IN': lambda x: (snap_date - x.max()).days, 'POLICYNO': 'count', 'PREMIUM': 'sum'}).rename(columns={'TGL_IN': 'RECENCY', 'POLICYNO': 'FREQ', 'PREMIUM': 'MONETARY'})
            
            # Ganti label status agar mudah dipahami
            rfm['STATUS'] = rfm.apply(lambda r: "PELANGGAN VIP" if r['RECENCY'] < 180 and r['MONETARY'] > 100000000 else ("AKTIF" if r['RECENCY'] < 180 else ("BERISIKO PINDAH" if r['RECENCY'] < 365 else "SUDAH TIDAK AKTIF")), axis=1)
            
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
                st.write("**Komposisi Status Nasabah**")
                fig_pie = px.pie(rfm['STATUS'].value_counts().reset_index(), values='count', names='STATUS', hole=0.5, 
                                 color='STATUS', color_discrete_map={'PELANGGAN VIP': C_SUCC, 'AKTIF': C_SEC, 'BERISIKO PINDAH': C_WARN, 'SUDAH TIDAK AKTIF': C_DANG})
                st.plotly_chart(make_chart(fig_pie), use_container_width=True)
            with c_rfm2:
                st.write("**⚠️ Peringatan Dini: Nasabah Aktif Yang Berpotensi Pergi**")
                st.caption("Sistem mendeteksi nasabah di bawah ini berpotensi pindah/berhenti berdasarkan frekuensi dan riwayat transaksi mereka. Segera hubungi mereka!")
                
                at_risk = rfm[rfm['STATUS'] == 'AKTIF'].sort_values('PROB_CHURN_%', ascending=False).head(10).reset_index()
                if not at_risk.empty:
                    st.dataframe(
                        at_risk[['INSURED_NAME', 'PROB_CHURN_%', 'FREQ', 'MONETARY']], 
                        use_container_width=True, hide_index=True, 
                        column_config={
                            "INSURED_NAME": st.column_config.TextColumn("Nama Nasabah", width="medium"), 
                            "PROB_CHURN_%": st.column_config.ProgressColumn("Risiko Pergi", min_value=0, max_value=100, format="%.1f%%"),
                            "FREQ": st.column_config.NumberColumn("Jml Transaksi"),
                            "MONETARY": st.column_config.NumberColumn("Total Transaksi", format="Rp %d")
                        }
                    )
                else:
                    st.success("Bagus! Saat ini tidak ada nasabah aktif yang berisiko tinggi untuk pergi.")
            
            st.markdown("---")
            st.write("**Peta Evaluasi Nasabah**")
            st.info("💡 **Cara Membaca:** Semakin titiknya berada di bawah (jarak transaksi dekat) dan di atas (sering bertransaksi), maka nasabah tersebut semakin berharga untuk kita.")
            
            rfm_plot = rfm.reset_index().copy()
            rfm_plot['MONETARY_SIZE'] = rfm_plot['MONETARY'].abs() + 1 
            
            fig_2d = px.scatter(
                rfm_plot, x='RECENCY', y='FREQ', size='MONETARY_SIZE', color='STATUS', hover_name='INSURED_NAME',
                hover_data={'MONETARY': ':,.0f', 'MONETARY_SIZE': False}, opacity=0.8, 
                color_discrete_map={'PELANGGAN VIP': C_SUCC, 'AKTIF': C_SEC, 'BERISIKO PINDAH': C_WARN, 'SUDAH TIDAK AKTIF': C_DANG},
                labels={'RECENCY': 'Jarak Sejak Transaksi Terakhir (Hari)', 'FREQ': 'Total Frekuensi Transaksi'}
            )
            st.plotly_chart(make_chart(fig_2d), use_container_width=True)

        # PERUBAHAN: TAB OPERASIONAL DIGABUNG DENGAN MITRA
        with t5:
            st.subheader("Evaluasi Kinerja Operasional & Mitra")
            c_op1, c_op2 = st.columns([1, 2])
            with c_op1:
                st.write("**Kepatuhan Durasi Proses (SLA)**")
                st.caption("Target kita adalah selesai di bawah 2 hari kerja.")
                fig_sla = px.pie(df['STATUS_SLA'].value_counts().reset_index(), values='count', names='STATUS_SLA', hole=0.5, color='STATUS_SLA', color_discrete_map={'AMAN':C_SUCC, 'TERLAMBAT':C_DANG})
                st.plotly_chart(make_chart(fig_sla), use_container_width=True)
                
            with c_op2:
                st.write("**Kinerja Kecepatan Tim Admin**")
                perf = df.groupby('INPUT_NAME').agg(PREMIUM=('PREMIUM', 'sum'), SLA_HARI=('SLA_HARI', 'mean'), POLICYNO=('POLICYNO', 'count')).reset_index().sort_values('POLICYNO', ascending=False).head(10)
                fig_bar_admin = px.bar(perf, x='POLICYNO', y='INPUT_NAME', orientation='h', text_auto=True, color='SLA_HARI', color_continuous_scale='RdYlGn_r', labels={'POLICYNO': 'Jumlah Dokumen', 'SLA_HARI': 'Rata-rata Hari Proses'})
                fig_bar_admin.update_layout(yaxis={'categoryorder':'total ascending', 'title': ''}, xaxis={'title': ''})
                st.plotly_chart(make_chart(fig_bar_admin), use_container_width=True)
                
            st.markdown("---")
            st.write("**Top 10 Mitra Agen & Broker Penyumbang Terbesar**")
            df_broker = df.groupby('MO_NAME').agg({'PREMIUM':'sum', 'POLICYNO':'count'}).reset_index()
            df_broker = df_broker[df_broker['PREMIUM'] > 0].sort_values('PREMIUM', ascending=False).head(10)
            
            fig_broker = px.bar(df_broker, x='PREMIUM', y='MO_NAME', orientation='h', color='PREMIUM', color_continuous_scale='Blues')
            fig_broker.update_traces(texttemplate='Rp %{x:,.0f}', textposition='inside', insidetextanchor='middle', textfont=dict(color='white', size=13))
            fig_broker.update_layout(yaxis={'categoryorder':'total ascending', 'title': ''}, xaxis={'visible': False})
            st.plotly_chart(make_chart(fig_broker), use_container_width=True)

        with t6:
            st.subheader("Pusat Unduh Data Mentah")
            st.write("Gunakan menu ini jika Anda membutuhkan data mentah untuk diperiksa di Excel.")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("📥 UNDUH DATA KE EXCEL (CSV)", data=df.to_csv(index=False).encode('utf-8'), file_name="Data_Lengkap_ACABogor.csv", mime="text/csv")
