import streamlit as st
import hashlib

def check_password():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        # Tampilan login yang lebih bersih dan elegan
        st.markdown("<h1 style='text-align: center; color: #0F172A;'>PORTAL EKSEKUTIF ACA BOGOR</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #475569;'>Sistem diamankan. Silakan masukkan kredensial Anda untuk mengakses dashboard.</p>", unsafe_allow_html=True)
        
        st.write("")
        st.write("")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("login_form"):
                user = st.text_input("Username")
                pwd = st.text_input("Password", type="password")
                submit = st.form_submit_button("Masuk ke Dashboard", use_container_width=True)
                
                if submit:
                    hashed_pwd = hashlib.sha256(pwd.encode()).hexdigest()
                    # Hash yang benar untuk 'admin123'
                    correct_hash = "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9" 
                    
                    if user == 'admin' and hashed_pwd == correct_hash:
                        st.session_state['logged_in'] = True
                        st.rerun()
                    else:
                        st.error("❌ Username atau Password salah! Pastikan Anda memasukkan data yang benar.")
        st.stop() # Hentikan eksekusi script selanjutnya jika belum login
