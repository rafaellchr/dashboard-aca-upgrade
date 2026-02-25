import streamlit as st
import pandas as pd
import glob
import os

@st.cache_data(ttl=3600) 
def load_data_hybrid(uploaded_files=None, date_mode="AUTO"):
    dfs = []
    parquet_found = False
    
    # 1. BACA PARQUET
    files_parquet = glob.glob(os.path.join("data_produksi", "*.parquet"))
    if files_parquet:
        for f in files_parquet:
            try:
                df_t = pd.read_parquet(f)
                df_t.columns = [str(c).strip().upper() for c in df_t.columns]
                dfs.append(df_t)
                parquet_found = True
            except Exception as e:
                st.error(f"Gagal baca parquet: {e}")

    # 2. BACA CSV JIKA TIDAK ADA PARQUET
    if not parquet_found:
        files_local = glob.glob(os.path.join("data_produksi", "*.csv"))
        for f in files_local:
            try:
                df_t = pd.read_csv(f, sep=None, engine='python', encoding='utf-8-sig', dtype=str, on_bad_lines='skip')
                df_t.columns = [str(c).strip().upper() for c in df_t.columns]
                dfs.append(df_t)
            except: pass

    # 3. BACA FILE UPLOAD
    if uploaded_files:
        for f in uploaded_files:
            try:
                if f.name.endswith('.parquet'):
                    df_t = pd.read_parquet(f)
                else:
                    df_t = pd.read_csv(f, sep=None, engine='python', encoding='utf-8-sig', dtype=str, on_bad_lines='skip')
                df_t.columns = [str(c).strip().upper() for c in df_t.columns]
                dfs.append(df_t)
            except: pass
            
    if not dfs: return None, "NO_DATA"
    df = pd.concat(dfs, ignore_index=True)

    if 'DATE_INPUT' not in df.columns: return None, "MISSING_COL"

    # --- [IMPROVEMENT KRUSIAL]: PENANGANAN EXCEL SERIAL DATE ---
    # Fungsi ini akan mengecek apakah tanggal berupa angka (misal: 45688.0) atau string biasa
    def parse_smart_date(series):
        # Coba ubah ke angka dulu
        num_dates = pd.to_numeric(series, errors='coerce')
        mask_num = num_dates.notna()
        
        # Untuk yang bukan angka, baca secara normal
        dates = pd.to_datetime(series, errors='coerce')
        
        # Untuk yang berupa angka (Excel Format), konversi dari hitungan hari sejak 1899
        dates.loc[mask_num] = pd.to_datetime(num_dates[mask_num], origin='1899-12-30', unit='D')
        return dates

    df['TGL_IN'] = parse_smart_date(df['DATE_INPUT'])
    
    if 'USER_APPROVE_DATE' in df.columns:
        df['TGL_APP'] = parse_smart_date(df['USER_APPROVE_DATE'])
        df['TGL_APP'] = df['TGL_APP'].fillna(df['TGL_IN'])
    else:
        df['TGL_APP'] = df['TGL_IN']
        
    if 'EXPIRY' in df.columns:
        df['TGL_EXP'] = parse_smart_date(df['EXPIRY'])
    
    # --- PEMBERSIHAN ANGKA ---
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
    
    # Memasukkan MO_NAME (Nama Agen) agar bisa dipakai di visualisasi
    for c in ['SEGMENT', 'TOC_DESCRIPTION', 'INPUT_NAME', 'INSURED_NAME', 'BRANCH', 'MO_NAME']:
        if c not in df.columns: df[c] = "UNKNOWN"
        else: df[c] = df[c].fillna("UNKNOWN").str.upper().str.strip()
        
    return df, None