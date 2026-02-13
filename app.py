import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import glob
import os

# --- 1. KONFIGURASI TAMPILAN (CLEAN PROFESSIONAL) ---
st.set_page_config(
    page_title="ACA BOGOR: EXECUTIVE DASHBOARD", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# CSS: TAMPILAN BERSIH & FORMAL
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
    .stTabs [data-baseweb="tab"] {
        font-size: 14px;
        font-weight: 700;
        color: #475569;
    }
</style>
""", unsafe_allow_html=True)

# WARNA TEMA
C_PRIM = "#2563eb" # Biru ACA
C_DANG = "#ef4444" # Merah Alert
C_SUCC = "#22c55e" # Hijau Aman
C_WARN = "#f59e0b" # Kuning Warning

# FUNGSI PEMBERSIH CHART
def make_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(family="Arial", size=11, color="#334155"),
        xaxis=dict(showgrid=False, linecolor="#cbd5e1"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
    )
    return fig

# --- 2. ENGINE DATA (OPTIMAL: PARQUET PRIORITY) ---
def load_data_hybrid(uploaded_files=None, date_mode="AUTO"):
    dfs = []
    parquet_found = False
    
    # 1. CEK FILE PARQUET (PRIORITAS UTAMA - LEBIH CEPAT)
    files_parquet = glob.glob(os.path.join("data_produksi", "*.parquet"))
    if files_parquet:
        for f in files_parquet:
            try:
                # Baca parquet
                df_t = pd.read_parquet(f)
                # Pastikan semua jadi string dulu biar aman
                df_t = df_t.astype(str)
                # Rapikan nama kolom
                df_t.columns = [str(c).strip().upper() for c in df_t.columns]
                dfs.append(df_t)
                parquet_found = True
            except Exception as e:
                st.error(f"Gagal baca parquet: {e}")

    # 2. CEK FILE CSV (HANYA JIKA PARQUET TIDAK ADA)
    # Ini mencegah data double jika di laptop ada CSV & Parquet sekaligus
    if not parquet_found:
        files_local = glob.glob(os.path.join("data_produksi", "*.csv"))
        for f in files_local:
            try:
                df_t = pd.read_csv(f, sep=None, engine='python', encoding='utf-8-sig', dtype=str, on_bad_lines='skip')
                df_t.columns = [str(c).strip().upper() for c in df_t.columns]
                dfs.append(df_t)
            except: pass

    # 3. CEK FILE UPLOAD (SELALU DI-LOAD JIKA ADA)
    if uploaded_files:
        for f in uploaded_files:
            try:
                if f.name.endswith('.parquet'):
                    df_t = pd.read_parquet(f)
                    df_t = df_t.astype(str)
                else:
                    df_t = pd.read_csv(f, sep=None, engine='python', encoding='utf-8-sig', dtype=str, on_bad_lines='skip')
                
                df_t.columns = [str(c).strip().upper() for c in df_t.columns]
                dfs.append(df_t)
            except: pass
            
    if not dfs: return None, "NO_DATA"
    
    df = pd.concat(dfs, ignore_index=True)

    # --- PEMBERSIHAN DATA ---
    if 'DATE_INPUT' not in df.columns: return None, "MISSING_COL"

    col_date = 'DATE_INPUT'
    df[col_date] = df[col_date].astype(str).str.split().str[0] 

    if date_mode == "INDO (Hari/Bulan)":
        df['TGL_IN'] = pd.to_datetime(df[col_date], dayfirst=True, errors='coerce')
    elif date_mode == "US (Bulan/Hari)":
        df['TGL_IN'] = pd.to_datetime(df[col_date], dayfirst=False, errors='coerce')
    else: 
        d1 = pd.to_datetime(df[col_date], format='%m/%d/%Y', errors='coerce')
        d2 = pd.to_datetime(df[col_date], format='%d/%m/%Y', errors='coerce')
        d3 = pd.to_datetime(df[col_date], errors='coerce')
        df['TGL_IN'] = d1.fillna(d2).fillna(d3)

    if 'USER_APPROVE_DATE' in df.columns:
        df['USER_APPROVE_DATE'] = df['USER_APPROVE_DATE'].astype(str).str.split().str[0]
        if date_mode == "INDO (Hari/Bulan)":
            df['TGL_APP'] = pd.to_datetime(df['USER_APPROVE_DATE'], dayfirst=True, errors='coerce')
        elif date_mode == "US (Bulan/Hari)":
            df['TGL_APP'] = pd.to_datetime(df['USER_APPROVE_DATE'], dayfirst=False, errors='coerce')
        else:
            da1 = pd.to_datetime(df['USER_APPROVE_DATE'], format='%m/%d/%Y', errors='coerce')
            da_auto = pd.to_datetime(df['USER_APPROVE_DATE'], errors='coerce')
            df['TGL_APP'] = da1.fillna(da_auto)
        df['TGL_APP'] = df['TGL_APP'].fillna(df['TGL_IN'])
    else:
        df['TGL_APP'] = df['TGL_IN']
    
    def clean_num(col):
        if col in df.columns:
            val = df[col].astype(str).str.replace(',', '', regex=False).str.replace('Rp', '', regex=False).str.strip()
            return pd.to_numeric(val, errors='coerce').fillna(0)
        return 0

    df['PREMIUM'] = clean_num('PREMIUM_GROSS')
    df['TSI_VAL'] = clean_num('TSI' if 'TSI' in df.columns else 'TSI_OC')
    
    df['RATE_PCT'] = 0.0
    mask = df['TSI_VAL'] > 0
    df.loc[mask, 'RATE_PCT'] = (df.loc[mask, 'PREMIUM'] / df.loc[mask, 'TSI_VAL']) * 100
    
    df['SLA_HARI'] = (df['TGL_APP'] - df['TGL_IN']).dt.days
    df['STATUS_SLA'] = df['SLA_HARI'].apply(lambda x: "ON TRACK" if x <= 2 else "DELAYED")
    
    df = df.dropna(subset=['TGL_IN'])
    
    df['TAHUN'] = df['TGL_IN'].dt.year.astype(str)
    df['BULAN_NUM'] = df['TGL_IN'].dt.month
    df['BULAN_NAMA'] = df['TGL_IN'].dt.strftime('%B').str.upper()
    
    for c in ['SEGMENT', 'TOC_DESCRIPTION', 'INPUT_NAME', 'INSURED_NAME']:
        if c not in df.columns: df[c] = "UNKNOWN"
        else: df[c] = df[c].fillna("UNKNOWN").str.upper().str.strip()
        
    return df, None

# --- 3. UI DASHBOARD UTAMA ---

st.sidebar.header("CONTROL PANEL")
date_option = st.sidebar.radio("Format Tanggal:", ["AUTO (Deteksi)", "INDO (Hari/Bulan)", "US (Bulan/Hari)"], index=0)
uploaded_files = st.sidebar.file_uploader("Upload CSV Tambahan", type=['csv', 'parquet'], accept_multiple_files=True)

df_raw, error_msg = load_data_hybrid(uploaded_files, date_option)

if df_raw is not None:
    years = sorted(df_raw['TAHUN'].unique())
    sel_years = st.sidebar.multiselect("PILIH TAHUN:", years, default=years)
    df = df_raw[df_raw['TAHUN'].isin(sel_years)]
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"TOTAL DATA: {len(df):,} Transaksi")
    
    st.title("EXECUTIVE DASHBOARD")
    st.caption(f"Monitoring Data: {', '.join(sel_years)}")
    
    if len(df) > 0:
        last_date = df['TGL_IN'].max()
        curr_month_df = df[(df['TGL_IN'].dt.year == last_date.year) & (df['TGL_IN'].dt.month == last_date.month)]
        prev_date = last_date - pd.DateOffset(months=1)
        prev_month_df = df[(df['TGL_IN'].dt.year == prev_date.year) & (df['TGL_IN'].dt.month == prev_date.month)]
        
        curr_omset = curr_month_df['PREMIUM'].sum()
        prev_omset = prev_month_df['PREMIUM'].sum()
        growth = ((curr_omset - prev_omset)/prev_omset * 100) if prev_omset > 0 else 0
        nama_bulan = last_date.strftime('%B %Y').upper()
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"OMSET {nama_bulan}", f"{curr_omset/1e9:.2f} M", f"{growth:.1f}% vs Lalu")
        k2.metric("TOTAL OMSET (YTD)", f"{df['PREMIUM'].sum()/1e9:.2f} M", "Total Filter")
        k3.metric("KECEPATAN (SLA)", f"{df['SLA_HARI'].mean():.1f} HARI", "Target < 2")
        k4.metric("TOTAL POLIS", f"{len(df):,}", "Transaksi")

        st.markdown("---")

        t1, t2, t3, t4, t5, t6 = st.tabs(["TREN BISNIS", "PRODUK", "NASABAH (CRM)", "AI INTELLIGENCE", "OPERASIONAL", "DATA"])
        
        # 1. TREN
        with t1:
            c1, c2 = st.columns([2,1])
            with c1:
                st.subheader("TREN PENDAPATAN (MULTI-YEAR)")
                trend = df.groupby(['TAHUN', 'BULAN_NUM', 'BULAN_NAMA'])['PREMIUM'].sum().reset_index().sort_values(['TAHUN', 'BULAN_NUM'])
                fig = px.line(trend, x='BULAN_NAMA', y='PREMIUM', color='TAHUN', markers=True, color_discrete_sequence=px.colors.qualitative.Bold)
                st.plotly_chart(make_chart(fig), use_container_width=True)
            with c2:
                st.subheader("POLA MUSIMAN")
                seas = df.groupby('BULAN_NAMA')['PREMIUM'].mean().reindex(['JANUARY','FEBRUARY','MARCH','APRIL','MAY','JUNE','JULY','AUGUST','SEPTEMBER','OCTOBER','NOVEMBER','DECEMBER']).reset_index()
                fig_s = px.bar(seas, x='BULAN_NAMA', y='PREMIUM', color='PREMIUM')
                st.plotly_chart(make_chart(fig_s), use_container_width=True)
            
            st.markdown("---")
            st.subheader("GROWTH HEATMAP (Year-on-Year)")
            pivot_heat = df.groupby(['TAHUN', 'BULAN_NUM', 'BULAN_NAMA'])['PREMIUM'].sum().reset_index().sort_values('BULAN_NUM')
            fig_heat = px.density_heatmap(pivot_heat, x='TAHUN', y='BULAN_NAMA', z='PREMIUM', color_continuous_scale='Blues', text_auto='.2s')
            st.plotly_chart(make_chart(fig_heat), use_container_width=True)

        # 2. PRODUK
        with t2:
            c_p1, c_p2 = st.columns([2,1])
            with c_p1:
                st.subheader("PETA DOMINASI PRODUK")
                df_pos = df[df['PREMIUM']>0]
                if not df_pos.empty:
                    fig_tree = px.treemap(df_pos, path=[px.Constant("TOTAL"), 'SEGMENT', 'TOC_DESCRIPTION'], values='PREMIUM', color='SEGMENT', color_discrete_sequence=px.colors.qualitative.Prism)
                    st.plotly_chart(make_chart(fig_tree), use_container_width=True)
            with c_p2:
                st.subheader("TOP 5 PRODUK")
                top_prod = df.groupby('TOC_DESCRIPTION')['PREMIUM'].sum().reset_index().sort_values('PREMIUM', ascending=False).head(5)
                fig_bar = px.bar(top_prod, x='PREMIUM', y='TOC_DESCRIPTION', orientation='h', text_auto='.2s')
                fig_bar.update_traces(marker_color=C_PRIM)
                st.plotly_chart(make_chart(fig_bar), use_container_width=True)
            
            st.markdown("---")
            st.subheader("ANALISIS TICKET SIZE (Distribusi Harga Polis)")
            fig_hist = px.histogram(df[df['PREMIUM']>0], x="PREMIUM", nbins=50, color_discrete_sequence=[C_WARN], title="Sebaran Nilai Premi per Transaksi")
            st.plotly_chart(make_chart(fig_hist), use_container_width=True)

        # 3. NASABAH
        with t3:
            st.subheader("ANALISIS KESEHATAN NASABAH (RFM)")
            snap_date = df['TGL_IN'].max() + timedelta(days=1)
            rfm = df.groupby('INSURED_NAME').agg({'TGL_IN': lambda x: (snap_date - x.max()).days, 'POLICYNO': 'count', 'PREMIUM': 'sum'}).rename(columns={'TGL_IN': 'RECENCY', 'POLICYNO': 'FREQ', 'PREMIUM': 'MONETARY'})
            def segmen_rfm(r):
                if r['RECENCY'] < 180 and r['MONETARY'] > 100000000: return "CHAMPIONS (VIP)"
                if r['RECENCY'] < 180: return "ACTIVE"
                if r['RECENCY'] < 365: return "RISIKO CHURN"
                return "SLEEPING (DORMAN)"
            rfm['STATUS'] = rfm.apply(segmen_rfm, axis=1)
            
            c_rfm1, c_rfm2 = st.columns([1,2])
            with c_rfm1:
                cnt = rfm['STATUS'].value_counts().reset_index()
                fig_pie = px.pie(cnt, values='count', names='STATUS', hole=0.5, color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(make_chart(fig_pie), use_container_width=True)
            with c_rfm2:
                st.write("**TOP NASABAH VIP (CHAMPIONS):**")
                st.dataframe(rfm[rfm['STATUS']=="CHAMPIONS (VIP)"].sort_values('MONETARY', ascending=False).head(50), use_container_width=True)
            
            st.markdown("---")
            st.subheader("HUKUM PARETO (80/20 Rule)")
            pareto_df = rfm.sort_values(by='MONETARY', ascending=False).reset_index()
            pareto_df['CUMULATIVE_PREMIUM'] = pareto_df['MONETARY'].cumsum()
            pareto_df['CUMULATIVE_PERCENT'] = 100 * pareto_df['CUMULATIVE_PREMIUM'] / pareto_df['MONETARY'].sum()
            pareto_df['CLIENT_RANK'] = pareto_df.index + 1
            pareto_viz = pareto_df.head(100)
            
            fig_pareto = go.Figure()
            fig_pareto.add_trace(go.Bar(x=pareto_viz['CLIENT_RANK'], y=pareto_viz['MONETARY'], name='Premi Nasabah', marker_color=C_PRIM))
            fig_pareto.add_trace(go.Scatter(x=pareto_viz['CLIENT_RANK'], y=pareto_viz['CUMULATIVE_PERCENT'], name='Kumulatif %', yaxis='y2', line=dict(color=C_DANG, width=3)))
            fig_pareto.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 110], title="Persentase Kumulatif"), xaxis=dict(title="Peringkat Nasabah (Top 100)"), title="Kontribusi Nasabah Terbesar (Top Clients)")
            st.plotly_chart(make_chart(fig_pareto), use_container_width=True)

        # 4. AI INTELLIGENCE
        with t4:
            st.markdown("### AI INTELLIGENCE CENTER")
            
            st.info("**1. AI OPPORTUNITY FINDER (Peluang Cross-Selling)** - *Produk apa yang sering dibeli bersamaan?*")
            
            basket = df.groupby(['INSURED_NAME', 'TOC_DESCRIPTION'])['POLICYNO'].count().unstack().fillna(0)
            basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
            
            if len(basket_sets.columns) > 1:
                co_matrix = basket_sets.T.dot(basket_sets)
                np.fill_diagonal(co_matrix.values, 0)
                
                stacked = co_matrix.stack()
                stacked.index.names = ['Product A', 'Product B'] 
                pairs = stacked.reset_index(name='Count')
                
                pairs = pairs[pairs['Count'] > 0]
                
                pairs['sorted_key'] = pairs.apply(lambda x: tuple(sorted([x['Product A'], x['Product B']])), axis=1)
                pairs = pairs.drop_duplicates(subset='sorted_key')
                
                top_pairs = pairs.sort_values('Count', ascending=False).head(10)
                
                if not top_pairs.empty:
                    top_pairs['Pair Name'] = top_pairs['Product A'] + " + " + top_pairs['Product B']
                    
                    c_ai_1, c_ai_2 = st.columns([2, 1])
                    
                    with c_ai_1:
                        fig_pairs = px.bar(top_pairs, x='Count', y='Pair Name', orientation='h',
                                             title="TOP 10 KOMBINASI PRODUK (Sering Dibeli Bersama)",
                                             text_auto=True, color='Count', color_continuous_scale='Greens')
                        fig_pairs.update_layout(yaxis=dict(autorange="reversed")) 
                        st.plotly_chart(make_chart(fig_pairs), use_container_width=True)
                    
                    with c_ai_2:
                        st.write("**REKOMENDASI AI:**")
                        best_pair = top_pairs.iloc[0]
                        st.success(f"Peluang Terbesar: Jika nasabah beli **{best_pair['Product A']}**, tawarkan **{best_pair['Product B']}**.")
                        st.caption("Data menunjukkan kombinasi ini paling sering terjadi.")
                else:
                    st.warning("Belum cukup data transaksi silang (Cross-Selling) untuk ditampilkan.")
            else:
                st.warning("Variasi produk belum cukup untuk analisis Cross-Selling.")

            st.markdown("---")
            
            st.error("**2. AI AUDITOR (Deteksi Anomali)** - *Mendeteksi transaksi dengan Rate Premi tidak wajar*")
            
            stats = df[df['RATE_PCT'] > 0].groupby('TOC_DESCRIPTION')['RATE_PCT'].agg(['mean', 'std']).reset_index()
            df_risk = pd.merge(df, stats, on='TOC_DESCRIPTION', how='left')
            
            df_risk['Z_SCORE'] = (df_risk['RATE_PCT'] - df_risk['mean']) / df_risk['std']
            anomalies = df_risk[(df_risk['Z_SCORE'].abs() > 2) & (df_risk['PREMIUM'] > 1000000)].sort_values('Z_SCORE')
            
            c_an1, c_an2 = st.columns([2,1])
            with c_an1:
                fig_anom = px.scatter(df_risk, x="PREMIUM", y="RATE_PCT", color="TOC_DESCRIPTION", 
                                      hover_data=['INSURED_NAME'], title="Sebaran Rate vs Premium (Titik Jauh = Anomali)")
                st.plotly_chart(make_chart(fig_anom), use_container_width=True)
            with c_an2:
                st.write(f"**DITEMUKAN: {len(anomalies)} TRANSAKSI ANOMALI**")
                st.caption("Transaksi ini memiliki Rate yang menyimpang jauh dari rata-rata produk sejenis.")
                st.dataframe(anomalies[['INSURED_NAME', 'TOC_DESCRIPTION', 'PREMIUM', 'RATE_PCT', 'Z_SCORE']].head(10), hide_index=True)

        # 5. OPERASIONAL
        with t5:
            c_op1, c_op2 = st.columns([1, 2])
            with c_op1:
                st.subheader("KEPATUHAN SLA")
                sla_counts = df['STATUS_SLA'].value_counts().reset_index()
                fig_sla = px.pie(sla_counts, values='count', names='STATUS_SLA', hole=0.5, color='STATUS_SLA', color_discrete_map={'ON TRACK':C_SUCC, 'DELAYED':C_DANG})
                st.plotly_chart(make_chart(fig_sla), use_container_width=True)
            with c_op2:
                st.subheader("BEBAN KERJA ADMIN")
                perf = df.groupby('INPUT_NAME').agg({'PREMIUM':'sum', 'SLA_HARI':'mean', 'POLICYNO':'count'}).reset_index()
                perf['SIZE_ABS'] = perf['PREMIUM'].abs() 
                fig_bub = px.scatter(perf, x='POLICYNO', y='SLA_HARI', size='SIZE_ABS', color='INPUT_NAME', hover_name='INPUT_NAME', hover_data={'SIZE_ABS':False, 'PREMIUM':True})
                fig_bub.add_hline(y=2, line_dash="dot", annotation_text="BATAS SLA")
                st.plotly_chart(make_chart(fig_bub), use_container_width=True)

        # 6. DATA
        with t6:
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("DOWNLOAD DATA CSV", data=csv, file_name="dashboard_aca_export.csv", mime="text/csv")

    else:
        st.warning("Data Kosong. Cek filter tahun atau format tanggal.")
else:
    st.info("Silakan masukkan file CSV/Parquet ke folder 'data_produksi'.")