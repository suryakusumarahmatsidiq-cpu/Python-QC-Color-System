import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================================
# 1. KONFIGURASI SYSTEM
# ==========================================================
st.set_page_config(page_title="Monitoring data historikal", layout="wide", page_icon="🛡️")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_NAME = "DATA_DIKI.xlsx"
FILE_PATH = os.path.join(BASE_DIR, FILE_NAME)

st.markdown("""
    <style>
    .stButton>button {width: 100%; border-radius: 5px; height: 3em; font-weight: bold;}
    .success {color: green;}
    .warning {color: orange;}
    .danger {color: red;}
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. UTILITY FUNCTIONS (UPDATED: CLEANER & FILLER)
# ==========================================================
def load_data():
    if not os.path.exists(FILE_PATH):
        st.error(f"❌ File '{FILE_NAME}' tidak ditemukan di folder script!")
        return None, None, None
    try:
        # 1. Load History
        df_hist = pd.read_excel(FILE_PATH, sheet_name="history_log").fillna(0)
        df_hist.columns = [str(c).strip() for c in df_hist.columns]
        
        if "Final_DE" in df_hist.columns:
            if df_hist["Final_DE"].dtype == object:
                df_hist["Final_DE"] = df_hist["Final_DE"].astype(str).str.replace(',', '.')
            df_hist["Final_DE"] = pd.to_numeric(df_hist["Final_DE"], errors='coerce').fillna(10.0)

        # 2. Load Recipe
        df_recipe = pd.read_excel(FILE_PATH, sheet_name="product_recipe").fillna(0)
        df_recipe.columns = [str(c).strip() for c in df_recipe.columns]

        # 3. Load Lab (AUTO-DETECT & CLEAN)
        try:
            xl = pd.ExcelFile(FILE_PATH)
            # Cari sheet yang namanya mirip 'lab' atau 'strength'
            sheet_lab = next((s for s in xl.sheet_names if any(x in s.lower() for x in ["lab", "str", "qc"])), "lab_strength")
            
            df_lab = pd.read_excel(FILE_PATH, sheet_name=sheet_lab)
            
            # Deteksi Kolom
            col_nama_pasta = next((c for c in df_lab.columns if "nama" in str(c).lower() and "pasta" in str(c).lower()), None)
            col_nilai = next((c for c in df_lab.columns if "strength" in str(c).lower() or "str" in str(c).lower()), None)
            col_tgl = next((c for c in df_lab.columns if "tanggal" in str(c).lower()), None)

            if col_nama_pasta and col_nilai and col_tgl:
                # Bersihkan Simbol % dan ,
                df_lab[col_nilai] = df_lab[col_nilai].astype(str).str.replace('%', '', regex=False).str.replace(',', '.', regex=False)
                df_lab[col_nilai] = pd.to_numeric(df_lab[col_nilai], errors='coerce')
                
                # Format Tanggal
                df_lab[col_tgl] = pd.to_datetime(df_lab[col_tgl], dayfirst=True, errors='coerce')
                
                # PIVOT TABEL
                df_lab = df_lab.pivot_table(index=col_tgl, columns=col_nama_pasta, values=col_nilai).reset_index()
                df_lab = df_lab.rename(columns={col_tgl: "Tanggal"})
                
                # URUTKAN TANGGAL
                df_lab = df_lab.sort_values("Tanggal")
                
                # FILL DOWN: Kalau ada tanggal kosong, pakai nilai tanggal sebelumnya (Biar grafik nyambung!)
                df_lab = df_lab.ffill().fillna(100.0) # Kalau masih kosong, isi 100
            
            if "Tanggal" in df_lab.columns:
                df_lab["Tanggal"] = pd.to_datetime(df_lab["Tanggal"], dayfirst=True, errors='coerce')
                
        except Exception as e:
            st.warning(f"Warning Lab Data: {e}")
            df_lab = pd.DataFrame()

        return df_hist, df_recipe, df_lab
    except Exception as e:
        st.error(f"Gagal load Excel: {e}")
        return None, None, None

def get_strength_at_date(df_lab, pasta_name, target_date):
    if df_lab.empty: return 1.0
    col_name = next((c for c in df_lab.columns if pasta_name.lower() in str(c).lower()), None)
    if not col_name: return 1.0
    
    target_date = pd.to_datetime(target_date)
    valid_lab = df_lab.dropna(subset=['Tanggal'])
    
    # Cari data sampai tanggal produksi
    past_data = valid_lab[valid_lab['Tanggal'] <= target_date]
    
    # PENTING: Buang data yang kolom pastanya NaN (Kosong)
    past_data = past_data.dropna(subset=[col_name])
    
    if past_data.empty: return 1.0
    
    # Ambil yang terakhir
    val = float(past_data.sort_values('Tanggal').iloc[-1][col_name])
    
    # Handle persen/desimal
    if val > 5: return val / 100.0
    return val

def get_latest_strength(df_lab, pasta_name):
    if df_lab.empty: return 1.0
    col_name = next((c for c in df_lab.columns if pasta_name.lower() in str(c).lower()), None)
    if not col_name: return 1.0
    
    # Ambil data yang TIDAK NaN
    latest_data = df_lab.dropna(subset=[col_name])
    
    if latest_data.empty: return 1.0
    
    latest = latest_data.sort_values("Tanggal", ascending=False).iloc[0]
    val = float(latest[col_name])
    if val > 5: return val / 100.0
    return val

def remove_outliers(data_list):
    arr = np.array(data_list)
    if len(arr)<3: return arr
    mean = np.mean(arr); std = np.std(arr)
    if std==0: return arr
    return np.array([x for x in arr if (mean - 1.5*std <= x <= mean + 1.5*std)])

def calculate_confidence(data):
    if len(data)==0: return 0,0
    mean_val = np.mean(data); std_dev = np.std(data)
    if mean_val==0: return 0.0,0.0
    cv = (std_dev/mean_val)*100
    score = max(0, 100-(cv*2))
    return score, std_dev

# ==========================================================
# 3. SMART ENGINE (CORE LOGIC)
# ==========================================================
def smart_engine(hist_df, lab_df, product_name, col_prod_hist, pasta_names, tank_id, total_kg, mode="PRECISION"):
    DE_LIMIT = 0.8 if mode=="PRECISION" else 1.5
    MAX_HISTORY = 10 if mode=="PRECISION" else 20
    
    # Clean Final_DE
    hist_df["Final_DE"] = pd.to_numeric(hist_df["Final_DE"], errors='coerce').fillna(10.0)

    # 1. FILTER PRODUK
    clean_target = str(product_name).strip().lower()
    hist_df['__temp_prod_filter'] = hist_df[col_prod_hist].astype(str).str.strip().str.lower()
    tank_hist = hist_df[hist_df['__temp_prod_filter'] == clean_target]

    if tank_hist.empty:
        st.warning(f"⚠️ Tidak ada data history untuk produk '{product_name}'")
        return [0]*len(pasta_names), [0]*len(pasta_names)

    # 2. FILTER TANK
    col_tank = next((c for c in tank_hist.columns if "tank" in str(c).lower()), None)
    if col_tank:
        tank_hist_filtered = tank_hist[tank_hist[col_tank].astype(str) == str(tank_id)]
        if tank_hist_filtered.empty:
            st.warning(f"⚠️ Data Tank {tank_id} kosong. Menggunakan data global.")
        else:
            tank_hist = tank_hist_filtered

    # 3. FILTER DE
    filtered_hist = tank_hist[tank_hist["Final_DE"] <= DE_LIMIT]
    if filtered_hist.empty: 
        filtered_hist = tank_hist[tank_hist["Final_DE"] <= 2.5] 
        if filtered_hist.empty:
             filtered_hist = tank_hist 

    final_hist = filtered_hist.sort_values("Tanggal", ascending=True).tail(MAX_HISTORY)
    
    final_recipe = []
    conf_scores = []
    
    for pasta in pasta_names:
        if pasta not in final_hist.columns:
            final_recipe.append(0.0); conf_scores.append(0); continue
            
        pct_murni_list = []
        for idx, row in final_hist.iterrows():
            str_then = get_strength_at_date(lab_df, pasta, row['Tanggal'])
            kg_act = row[pasta]
            
            raw_batch = row.get('Total Batch (kg)', 0)
            try: tot_batch = float(raw_batch)
            except: tot_batch = 1000 
            if tot_batch <= 0: tot_batch = 1000 
            
            pct_murni = (kg_act * str_then / tot_batch) * 100
            if pct_murni > 0: pct_murni_list.append(pct_murni)
        
        clean_pcts = remove_outliers(pct_murni_list)
        if len(clean_pcts)==0: clean_pcts = pct_murni_list
        
        avg_murni = np.median(clean_pcts) if len(clean_pcts)>0 else 0
        conf, var = calculate_confidence(clean_pcts)
        
        str_now = get_latest_strength(lab_df, pasta)
        if str_now <= 0: str_now = 1.0
            
        kg_target = (avg_murni / 100 * total_kg) / str_now
        
        final_recipe.append(kg_target)
        conf_scores.append(conf)
        
    return final_recipe, conf_scores

# ==========================================================
# 4. DASHBOARD UI
# ==========================================================
df_hist, df_recipe, df_lab = load_data()
if df_hist is None: st.stop()

with st.sidebar:
    st.header("🎛️ INPUT DATA")
    col_prod_recipe = st.selectbox("1. Kolom Produk (Recipe):", df_recipe.columns)
    col_prod_hist = st.selectbox("2. Kolom Produk (History):", df_hist.columns)
    
    st.write("---")
    st.header("🎛️ OPERASIONAL")
    mode_selected = st.radio("Mode:", ["PRECISION", "FAST"], help="Precision = Safety Factor Tinggi")
    
    produk_list = df_recipe[col_prod_recipe].unique()
    produk = st.selectbox("Pilih Produk", produk_list)
    
    col_tank_sb = next((c for c in df_hist.columns if "tank" in str(c).lower()), None)
    tank_list = df_hist[col_tank_sb].unique() if col_tank_sb else ['T-01','T-02']
    tank_id = st.selectbox("Pilih Tank", tank_list)
    
    total_kg = st.number_input("Total Batch (kg)", value=805.0, step=10.0)
    
    if st.button("🔄 Reset Form"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

st.title("🛡️ Monitoring data historikal CM dan semi pasta")
st.markdown(f"**Produk:** {produk} | **Tank:** {tank_id} | **Total:** {total_kg} kg")
st.write("---")

col_komp = next((c for c in df_recipe.columns if "komposisi" in str(c).lower()), None)
if col_komp:
    row_target = df_recipe[df_recipe[col_prod_recipe]==produk].iloc[0]
    str_komposisi = str(row_target[col_komp])
    pastas = [x.strip() for x in str_komposisi.replace('"','').split(',') if x.strip()]
else:
    st.error("Kolom 'Komposisi' tidak ditemukan!")
    st.stop()

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("💡 Rekomendasi Resep")
    
    if st.button("🚀 Hitung prediksi"):
        resep_kg, conf_scores = smart_engine(df_hist, df_lab, produk, col_prod_hist, pastas, tank_id, total_kg, mode_selected)
        
        st.session_state['resep'] = resep_kg
        st.session_state['conf'] = conf_scores
        st.session_state['pastas'] = pastas
        
        with st.expander(f"🕵️‍♂️ Lihat History Tank {tank_id}"):
             clean_prod = str(produk).strip().lower()
             df_hist['__debug_filter'] = df_hist[col_prod_hist].astype(str).str.strip().str.lower()
             col_tank_debug = next((c for c in df_hist.columns if "tank" in str(c).lower()), None)
             
             if col_tank_debug:
                 display_df = df_hist[(df_hist['__debug_filter'] == clean_prod) & (df_hist[col_tank_debug].astype(str) == str(tank_id))]
             else:
                 display_df = df_hist[df_hist['__debug_filter'] == clean_prod]
             
             if not display_df.empty:
                 st.dataframe(display_df.tail(5))
             else:
                 st.write("History kosong.")

    if 'resep' in st.session_state:
        resep_kg = st.session_state['resep']
        conf_scores = st.session_state['conf']
        pastas = st.session_state['pastas']
        
        if sum(resep_kg) == 0:
            st.warning("⚠️ Hasil 0. Cek kolom Produk.")
        else:
            data_table = []
            
            # --- LOGIKA SAFETY 80% ---
            SAFE_FACTOR_HIGH = 0.85
            SAFE_FACTOR_LOW = 0.70
            
            for i, p in enumerate(pastas):
                target = resep_kg[i]
                conf = conf_scores[i]
                
                factor = SAFE_FACTOR_HIGH
                status_txt = "🛡️ AMAN (85%)"
                
                if conf < 60:
                    factor = SAFE_FACTOR_LOW
                    status_txt = "🚧 RAGU (70%)"
                
                if mode_selected == "FAST":
                    factor += 0.05
                    status_txt = status_txt.replace("%)", "+5%)")

                tuang = target * factor
                tahan = target - tuang
                icon_conf = "🟢" if conf >= 60 else "🔴"
                
                data_table.append({
                    "Material": p,
                    "Target (kg)": f"{target:.3f}",
                    "TUANG (kg)": f"{tuang:.3f}",
                    "SISA (kg)": f"{tahan:.3f}",
                    "Status": status_txt,
                    "Conf": f"{icon_conf} {conf:.0f}%"
                })
                
            st.dataframe(pd.DataFrame(data_table), use_container_width=True)
            st.info("ℹ️ Target adalah hitungan murni. TUANG sudah dikurangi safety factor.")

            st.write("---")
            with st.form("save_form"):
                c1, c2 = st.columns(2)
                no_batch_input = c1.text_input("Nomor Batch")
                final_de_input = c2.number_input("Final DE Result", value=0.0)
                
                cols = st.columns(3)
                aktual_vals = {}
                for i, p in enumerate(pastas):
                    with cols[i % 3]:
                        val = st.text_input(f"{p} (kg)", value=f"{resep_kg[i]:.3f}")
                        aktual_vals[p] = float(val) if val else 0.0
                
                submit = st.form_submit_button("✅ SIMPAN KE HISTORY")
                if submit and no_batch_input:
                    new_data = {
                        "Tanggal": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "No Batch": no_batch_input,
                        "Kode Tank": tank_id,
                        col_prod_hist: produk, 
                        "Total Batch (kg)": total_kg,
                        "Final_DE": final_de_input
                    }
                    for p in pastas: new_data[p] = aktual_vals[p]
                    try:
                        df_new = pd.concat([df_hist.drop(columns=['__debug_filter'], errors='ignore'), pd.DataFrame([new_data])], ignore_index=True)
                        with pd.ExcelWriter(FILE_PATH, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                             if "history_log" in writer.book.sheetnames: writer.book.remove(writer.book["history_log"])
                             df_new.to_excel(writer, sheet_name="history_log", index=False)
                        st.success("Tersimpan!")
                    except Exception as e: st.error(f"Error Save: {e}")

with col2:
    st.subheader("📊 Analisa Lengkap")
    tab1, tab2, tab3 = st.tabs(["Trend Tank", "Komposisi", "🕵️‍♂️ Forensik"])
    
    clean_prod = str(produk).strip().lower()
    df_hist['__debug_filter'] = df_hist[col_prod_hist].astype(str).str.strip().str.lower()
    col_tank_chart = next((c for c in df_hist.columns if "tank" in str(c).lower()), None)
    
    if col_tank_chart:
         df_chart = df_hist[(df_hist['__debug_filter'] == clean_prod) & (df_hist[col_tank_chart].astype(str) == str(tank_id))].copy()
    else:
         df_chart = df_hist[df_hist['__debug_filter'] == clean_prod].copy()

    with tab1:
        if not df_chart.empty and 'Tanggal' in df_chart.columns:
            df_chart['Tanggal'] = pd.to_datetime(df_chart['Tanggal'])
            df_melt = df_chart.melt(id_vars=['Tanggal'], value_vars=[p for p in pastas if p in df_chart.columns], var_name='Pasta', value_name='Kg')
            fig = px.line(df_melt, x='Tanggal', y='Kg', color='Pasta', title=f"Trend {tank_id}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Trend kosong.")
            
    with tab2:
        if 'resep' in st.session_state:
             fig_pie = px.pie(values=resep_kg, names=pastas)
             st.plotly_chart(fig_pie, use_container_width=True)

    with tab3:
        st.write("#### ⚔️ Adu Data: Pemakaian vs Strength")
        pasta_pilihan = st.selectbox("Pilih Pasta:", pastas)
        
        df_forensik = df_hist[df_hist['__debug_filter'] == clean_prod].copy()
        
        if 'Tanggal' in df_forensik.columns and pasta_pilihan in df_forensik.columns and not df_forensik.empty:
            df_forensik['Tanggal'] = pd.to_datetime(df_forensik['Tanggal'])
            df_forensik = df_forensik.sort_values('Tanggal')
            
            avg_usage = df_forensik[pasta_pilihan].mean()
            if avg_usage == 0: avg_usage = 1
            usage_pct = (df_forensik[pasta_pilihan] / avg_usage) * 100
            
            strength_vals = []
            valid_strength = [] 
            for tgl in df_forensik['Tanggal']:
                v = get_strength_at_date(df_lab, pasta_pilihan, tgl)
                val_pct = v * 100
                strength_vals.append(val_pct)
                if val_pct != 100: valid_strength.append(val_pct)
            
            fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_dual.add_trace(go.Scatter(x=df_forensik['Tanggal'], y=usage_pct, name="DOSIS (Biru)", 
                                          line=dict(color='blue', width=4), mode='lines+markers',
                                          hovertemplate='Tanggal: %{x}<br>Dosis: %{y:.1f}%'), secondary_y=False)
            
            fig_dual.add_trace(go.Scatter(x=df_forensik['Tanggal'], y=strength_vals, name="KUALITAS (Merah)", 
                                          line=dict(color='red', width=3, dash='dot'), mode='lines+markers',
                                          hovertemplate='Tanggal: %{x}<br>Strength: %{y:.2f}%'), secondary_y=True)
            
            fig_dual.add_hline(y=100, line_dash="dash", line_color="green", secondary_y=False)

            if valid_strength:
                min_s = min(valid_strength) - 2
                max_s = max(valid_strength) + 2
            else:
                min_s, max_s = 95, 105
            
            fig_dual.update_layout(height=500, legend=dict(orientation="h", y=1.1, x=0.5))
            fig_dual.update_yaxes(title_text="Dosis (%)", secondary_y=False)
            fig_dual.update_yaxes(title_text="Kualitas (%)", secondary_y=True, range=[min_s, max_s], showgrid=False)
            
            st.plotly_chart(fig_dual, use_container_width=True)
        else:
            st.warning("Data tidak cukup.")