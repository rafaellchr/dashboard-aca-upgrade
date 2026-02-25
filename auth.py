import streamlit as st
import hashlib

def check_password():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        st.title("PORTAL LOGIN DASHBOARD")
        st.info("Sistem diamankan. Silakan masukkan kredensial Anda.")
        with st.form("login_form"):
            c_usr, c_pwd = st.columns(2)
            user = c_usr.text_input("Username")
            pwd = c_pwd.text_input("Password", type="password")
            submit = st.form_submit_button("Masuk ke Dashboard")
            
            if submit:
                hashed_pwd = hashlib.sha256(pwd.encode()).hexdigest()
                # [DIPERBAIKI] Ini adalah hash yang benar untuk 'admin123'
                correct_hash = "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9" 
                
                if user == 'admin' and hashed_pwd == correct_hash:
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("❌ Username atau Password salah! (Hint: admin / admin123)")
        st.stop() # Hentikan eksekusi script selanjutnya jika belum login