# -*- coding: utf-8 -*-
"""
İnşaat Proje Maliyeti Tahmin Aracı
Hafta 2'de kurduğumuz lineer regresyon modellerinin "gerçek kullanım ortamı".
Kod bilmeyen biri bile bu ekrandan tahmini maliyeti öğrenebilir.
"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Sayfa ayarları ve stil
# ------------------------------------------------------------------
st.set_page_config(
    page_title="İnşaat Maliyet Tahmin Aracı",
    page_icon="🏗️",
    layout="centered",
)

PRIMARY = "#1F5C99"
NAVY = "#0A2342"
AMBER = "#E8A838"
GREEN = "#2E7D32"
RED = "#C0392B"

st.markdown(f"""
<style>
    .main {{ background-color: #F4F7FA; }}
    .stApp header {{ background-color: transparent; }}
    h1 {{ color: {NAVY}; }}
    .app-header {{
        background-color: {NAVY};
        padding: 1.3rem 1.6rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
    }}
    .app-header h1 {{ color: white; margin: 0; font-size: 1.6rem; }}
    .app-header p {{ color: #D5E8F0; margin: 0.3rem 0 0 0; font-size: 0.95rem; }}
    .result-box {{
        background-color: {PRIMARY};
        color: white;
        padding: 1.4rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }}
    .result-box .value {{ font-size: 2.3rem; font-weight: 700; }}
    .result-box .label {{ font-size: 0.95rem; opacity: 0.85; }}
    .warn-box {{
        background-color: #FDEDEC;
        border: 1.5px solid {RED};
        color: {RED};
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        font-size: 0.92rem;
        margin-top: 0.6rem;
    }}
    .ok-box {{
        background-color: #EAF6EC;
        border: 1.5px solid {GREEN};
        color: {GREEN};
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        font-size: 0.92rem;
        margin-top: 0.6rem;
    }}
    footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <h1>🏗️ İnşaat Proje Maliyeti Tahmin Aracı</h1>
    <p>İnşaat Mühendisliğinde Yapay Zekâ Uygulamaları — Hafta 2 Lab Projesi</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Model ve metaveriyi yükle (Hücre 6'da kaydettiğimiz dosyalar)
# ------------------------------------------------------------------
@st.cache_resource
def yukle():
    basit = joblib.load("models/maliyet_modeli_basit.pkl")
    gelismis = joblib.load("models/maliyet_modeli_gelismis.pkl")
    with open("models/meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return basit, gelismis, meta

try:
    model_basit, model_gelismis, meta = yukle()
except FileNotFoundError:
    st.error(
        "Model dosyaları bulunamadı. Önce `python train_models.py` çalıştırarak "
        "modelleri eğitip `models/` klasörüne kaydedin."
    )
    st.stop()

# ------------------------------------------------------------------
# Kenar çubuğu: model seçimi + bilgi
# ------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Model Seçimi")
model_secimi = st.sidebar.radio(
    "Hangi modeli kullanmak istersiniz?",
    ["Basit Model (Hücre 5)", "Gelişmiş Model (Hücre 5 — Devam)"],
    help="Basit model sadece alan ve kat sayısını kullanır. Gelişmiş model "
         "zemin sınıfı ve inşaat yılını da ekler (bkz. Hafta 2 ders notu).",
)
gelismis_mi = model_secimi.startswith("Gelişmiş")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Bilgisi")
if gelismis_mi:
    m = meta["gelismis_model"]
    st.sidebar.metric("Test R²", f"{m['r2']:.3f}")
    st.sidebar.metric("Train R²", f"{m['train_r2']:.3f}")
    st.sidebar.metric("Test MAE", f"{m['mae']:,.0f} TL")
    st.sidebar.caption(
        "⚠️ Train R² ile Test R² arasındaki büyük fark, bu modelin "
        "**aşırı öğrenme (overfitting)** riski taşıdığını gösterir — "
        "Hafta 2'de birlikte incelediğimiz konu tam olarak budur."
    )
else:
    m = meta["basit_model"]
    st.sidebar.metric("Test R²", f"{m['r2']:.3f}")
    st.sidebar.metric("Test MAE", f"{m['mae']:,.0f} TL")
    st.sidebar.caption(
        "⚠️ R² negatif — bu, modelin sadece 2 değişkenle (alan, kat) "
        "yeterince açıklayıcı olmadığının işaretidir."
    )
st.sidebar.markdown("---")
st.sidebar.caption(f"Eğitim verisi: {meta['n_proje']} proje kaydı")

# ------------------------------------------------------------------
# Girdi formu
# ------------------------------------------------------------------
st.markdown("### Proje Bilgilerini Girin")

col1, col2 = st.columns(2)
with col1:
    alan_m2 = st.number_input(
        "Taban Alanı (m²)", min_value=50, max_value=10000, value=1200, step=50,
    )
with col2:
    kat_sayisi = st.number_input(
        "Kat Sayısı", min_value=1, max_value=40, value=8, step=1,
    )

zemin_sinifi = None
insaat_yili = None
if gelismis_mi:
    col3, col4 = st.columns(2)
    with col3:
        zemin_sinifi = st.selectbox("Zemin Sınıfı (TBDY'ye göre)", meta["zemin_siniflari"], index=0)
    with col4:
        insaat_yili = st.number_input(
            "İnşaat (Bitiş) Yılı", min_value=2015, max_value=2030,
            value=meta["insaat_yili"]["max"], step=1,
        )

# ------------------------------------------------------------------
# Ekstrapolasyon kontrolü (Hafta 2, "Modelin Sınırlarını Bilmek")
# ------------------------------------------------------------------
def araligin_disinda_mi(deger, anahtar):
    lo, hi = meta[anahtar]["min"], meta[anahtar]["max"]
    return deger < lo or deger > hi, lo, hi

uyarilar = []
disi, lo, hi = araligin_disinda_mi(alan_m2, "alan_m2")
if disi:
    uyarilar.append(f"**Alan** ({alan_m2:,} m²) eğitim verisinin aralığının ({lo:,}-{hi:,} m²) dışında.")
disi, lo, hi = araligin_disinda_mi(kat_sayisi, "kat_sayisi")
if disi:
    uyarilar.append(f"**Kat sayısı** ({kat_sayisi}) eğitim verisinin aralığının ({lo}-{hi}) dışında.")
if gelismis_mi:
    disi, lo, hi = araligin_disinda_mi(insaat_yili, "insaat_yili")
    if disi:
        uyarilar.append(f"**İnşaat yılı** ({insaat_yili}) eğitim verisinin aralığının ({lo}-{hi}) dışında.")

# ------------------------------------------------------------------
# Tahmin
# ------------------------------------------------------------------
if st.button("💰 Maliyeti Tahmin Et", type="primary", use_container_width=True):
    if gelismis_mi:
        row = {"alan_m2": alan_m2, "kat_sayisi": kat_sayisi, "insaat_yili": insaat_yili,
               "zemin_sinifi_B": 0, "zemin_sinifi_C": 0, "zemin_sinifi_D": 0}
        if zemin_sinifi != "A":
            row[f"zemin_sinifi_{zemin_sinifi}"] = 1
        X_yeni = pd.DataFrame([row])[meta["gelismis_model"]["features"]]
        tahmin = model_gelismis.predict(X_yeni)[0]
    else:
        X_yeni = pd.DataFrame([{"alan_m2": alan_m2, "kat_sayisi": kat_sayisi}])
        tahmin = model_basit.predict(X_yeni)[0]

    if uyarilar:
        st.markdown(f"""
        <div class="result-box" style="background-color:{RED};">
            <div class="value">{tahmin:,.0f} TL</div>
            <div class="label">Tahmini Toplam Maliyet — GÜVENİLİR DEĞİL</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
            '<div class="warn-box"><b>⚠️ Ekstrapolasyon uyarısı:</b> ' +
            " ".join(uyarilar) +
            " Model bu bölgede hiçbir şey öğrenmemiştir, tahmin güvenilir değildir.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"""
        <div class="result-box">
            <div class="value">{tahmin:,.0f} TL</div>
            <div class="label">Tahmini Toplam Maliyet</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
            '<div class="ok-box">✅ Girdi değerleri eğitim verisinin aralığı içinde — tahmin makul bir güvenilirlik taşıyor.</div>',
            unsafe_allow_html=True,
        )

    with st.expander("📐 Model bu tahmini nasıl hesapladı?"):
        if gelismis_mi:
            c = meta["gelismis_model"]["coefs"]
            st.markdown(f"""
Gelişmiş model, her özelliğin katsayısını (diğerleri sabitken) şu şekilde kullanır:

- Alan katsayısı: **{c['alan_m2']:,.0f} TL/m²**
- Kat katsayısı: **{c['kat_sayisi']:,.0f} TL/kat**
- Yıl katsayısı: **{c['insaat_yili']:,.0f} TL/yıl**
- Zemin B/C/D etkisi: **{c['zemin_sinifi_B']:,.0f}** / **{c['zemin_sinifi_C']:,.0f}** / **{c['zemin_sinifi_D']:,.0f}** TL (A zeminine göre farkı)
            """)
        else:
            b = meta["basit_model"]
            st.markdown(f"""
Basit model şu formülü kullanır:

**Maliyet = {b['alan_katsayisi']:,.0f} × Alan + {b['kat_katsayisi']:,.0f} × Kat + sabit**
            """)
        st.caption(
            "Not: Bu bir karar destek aracıdır, karar verici değil. Nihai kararı her zaman mühendis verir "
            "(bkz. Hafta 2 ders notu, Bölüm: Model Eğitildikten Sonra Nasıl Kullanılır?)."
        )

st.markdown("---")
st.caption(
    "Bu araç, Hafta 2 dersinde eğitilip Google Drive'a kaydedilen modelin "
    "kod yazmayan kullanıcılar için bir web arayüzüne taşınmış hâlidir. "
    "İnşaat Mühendisliğinde Yapay Zekâ Uygulamaları | 4. Sınıf | Güz Yarıyılı"
)
