"""Penguin Classifier: antarmuka offline dengan artefak hasil training nyata."""
from __future__ import annotations

import csv
import html
import io
import logging

import pandas as pd
import streamlit as st

from penguin_core import ROOT, FEATURES, LABELS, DISPLAY, read_json, load_artifacts, predict_frame

st.set_page_config(page_title="Penguin Classifier", page_icon="🐧", layout="wide")
logger=logging.getLogger(__name__)

st.markdown("<style>"+(ROOT / "assets" / "theme.css").read_text(encoding="utf-8")+"</style>",unsafe_allow_html=True)


@st.cache_resource
def resources(model_version, summary_version):
    # mtime sebagai cache key: training ulang tidak menyisakan model lama di cache.
    return load_artifacts()


def penguin_svg(label="Adelie"):
    accents={"Adelie":"#ed986d","Chinstrap":"#138795","Gentoo":"#7567ab"}
    color=accents.get(label,"#ed986d")
    return f'''<svg width="100" height="125" viewBox="0 0 120 150" xmlns="http://www.w3.org/2000/svg" aria-label="Ilustrasi dekoratif penguin">
    <ellipse cx="60" cy="139" rx="46" ry="8" fill="#cedfe3"/>
    <path d="M26 55 L6 106 L33 96 M94 55 L114 106 L87 96" fill="#173b4b"/>
    <ellipse cx="60" cy="77" rx="41" ry="64" fill="#173b4b"/>
    <ellipse cx="60" cy="91" rx="30" ry="45" fill="white"/>
    <ellipse cx="47" cy="48" rx="16" ry="18" fill="white"/><ellipse cx="73" cy="48" rx="16" ry="18" fill="white"/>
    <circle cx="48" cy="47" r="4" fill="#173b4b"/><circle cx="72" cy="47" r="4" fill="#173b4b"/>
    <path d="M51 59 L69 59 L60 71 Z" fill="{color}"/>
    <ellipse cx="43" cy="137" rx="15" ry="6" fill="{color}"/><ellipse cx="77" cy="137" rx="15" ry="6" fill="{color}"/>
    </svg>'''


st.markdown(f'''<div class="app-brandbar">
<div class="brand-icon">{penguin_svg()}</div>
<div><div class="brand-name">Penguin Classifier</div><div class="brand-subtitle">Kenali spesies lewat ukuran tubuh</div></div>
<div class="brand-badge">PALMER PENGUINS</div>
</div>''',unsafe_allow_html=True)


try:
    model_path=ROOT / "models" / "penguin_pipeline.joblib"
    summary_path=ROOT / "results" / "summary.json"
    model,summary=resources(model_path.stat().st_mtime_ns,summary_path.stat().st_mtime_ns)
except Exception as exc:
    logger.exception("Gagal memuat artefak prediksi")
    st.error("Layanan prediksi sedang tidak tersedia. Silakan coba lagi nanti atau hubungi pengelola aplikasi.")
    st.stop()

profile=read_json(ROOT / "results" / "data_profile.json")
samples=pd.read_csv(ROOT / "examples" / "contoh_berlabel.csv")
cv=pd.read_csv(ROOT / "results" / "cv_results.csv")
test=pd.read_csv(ROOT / "results" / "test_results.csv")
selected=summary["selected_model"]
selected_cv=cv[(cv.model==selected)&(cv.feature_set=="4 fitur")].iloc[0]
selected_test=test[test.model==selected].iloc[0]

# Sesi pengguna yang sedang membuka menu versi lama tetap dapat dipakai.
menus=["Prediksi","Cara menggunakan","Tentang model","Tentang dataset"]
previous_nav=st.session_state.get("navigation")
if previous_nav=="Model & evaluasi":
    st.session_state["navigation"]="Tentang model"
elif previous_nav is not None and previous_nav not in menus:
    st.session_state["navigation"]="Prediksi"

with st.sidebar:
    nav=st.radio("Menu aplikasi",menus,key="navigation")
    st.divider()
    st.markdown('''<div class="sidebar-label">LANGKAH PENGGUNA</div>
    <div class="sidebar-step"><span>1</span><div><strong>Isi pengukuran</strong><small>Gunakan satuan yang benar</small></div></div>
    <div class="sidebar-step"><span>2</span><div><strong>Klik prediksi</strong><small>Jalankan identifikasi spesies</small></div></div>
    <div class="sidebar-step"><span>3</span><div><strong>Baca hasil</strong><small>Lihat spesies dan skor model</small></div></div>
    <div class="sidebar-species"><strong>SPESIES YANG DAPAT DIKENALI</strong><p>Adelie</p><p>Chinstrap</p><p>Gentoo</p></div>''',unsafe_allow_html=True)
    st.caption("Siapkan ukuran tubuh dalam milimeter dan berat dalam gram.")


def show_single():
    st.title("Penguin apa yang sedang kita ukur?")
    st.write("Masukkan empat ukuran tubuh, lalu lihat prediksi spesiesnya.")
    st.markdown(f'''<div class="hero"><div><h2>Dari empat angka, kenali tiga spesies.</h2>
    <p>Adelie, Chinstrap, dan Gentoo dari dataset Palmer Penguins.</p>
    <span class="tag">{summary['n_source']} data sumber</span><span class="tag">3 spesies</span>
    <span class="tag">4 ukuran tubuh</span></div><div class="hero-illustration">{penguin_svg()}</div></div>''',unsafe_allow_html=True)
    single_tab,batch_tab=st.tabs(["Prediksi satu penguin","Batch CSV"])
    with single_tab:
        defaults=dict(zip(FEATURES,[39.1,18.7,181.0,3750.0]))
        for key,value in defaults.items():
            st.session_state.setdefault(key,value)

        def fill_sample(species=None):
            values=defaults if species is None else samples[samples.species==species].iloc[0].to_dict()
            for key in FEATURES:
                st.session_state[key]=float(values[key])
            st.session_state.pop("last_prediction",None)

        left,right=st.columns([1,1],gap="large")
        with left,st.container(border=True,key="measurement_card"):
            st.markdown('<div class="smallcaps">01 / INPUT PENGUKURAN</div>',unsafe_allow_html=True)
            st.subheader("Data penguin")
            buttons=st.columns(3)
            for col,species in zip(buttons,LABELS):
                col.button(f"Contoh {species}",on_click=fill_sample,args=(species,),width="stretch",help="Isi otomatis menggunakan contoh pengukuran dari dataset Palmer Penguins.")
            with st.form("single_form"):
                c1,c2=st.columns(2)
                with c1:
                    st.number_input("Panjang paruh (mm)",min_value=0.0,max_value=1000.0,step=.1,format="%.1f",key=FEATURES[0])
                    st.number_input("Panjang sirip (mm)",min_value=0.0,max_value=2000.0,step=1.0,format="%.1f",key=FEATURES[2])
                with c2:
                    st.number_input("Kedalaman paruh (mm)",min_value=0.0,max_value=1000.0,step=.1,format="%.1f",key=FEATURES[1])
                    st.number_input("Berat tubuh (gram)",min_value=0.0,max_value=100000.0,step=25.0,format="%.1f",key=FEATURES[3],help="3,75 kg = 3750 gram. Jangan isi dalam kilogram.")
                st.caption("Gunakan angka positif. Contoh: 3,75 kg ditulis sebagai 3750 gram.")
                submitted=st.form_submit_button("Prediksi spesies",type="primary",width="stretch")
            st.button("Reset form",on_click=fill_sample,width="stretch")
            if submitted:
                raw=pd.DataFrame([{c:st.session_state[c] for c in FEATURES}])
                try:
                    result,warnings=predict_frame(model,raw,summary)
                    st.session_state.last_prediction={"input":raw.to_dict("records")[0],"result":result.to_dict("records")[0],"warnings":warnings}
                except ValueError as exc:
                    st.session_state.pop("last_prediction",None)
                    st.error(str(exc))
        with right,st.container(border=True,key="result_card"):
            st.markdown('<div class="smallcaps">02 / HASIL IDENTIFIKASI</div>',unsafe_allow_html=True)
            st.subheader("Prediksi spesies")
            saved=st.session_state.get("last_prediction")
            if saved:
                current={c:st.session_state[c] for c in FEATURES}
                if current != saved["input"]:
                    st.warning("Hasil di bawah milik input terakhir yang dikirim. Klik Prediksi spesies untuk memperbarui.")
                record=saved["result"]
                label=record["prediksi"]
                st.markdown(f'<div class="species">{penguin_svg(label)}<div><p>HASIL MODEL {html.escape(selected)}</p><h2>{label}</h2><p>Skor tertinggi {record["skor_tertinggi"]:.1%}</p></div></div>',unsafe_allow_html=True)
                st.markdown('<div class="score-heading">Skor per spesies</div>',unsafe_allow_html=True)
                for species in model.classes_:
                    value=float(record[f"skor_{species}"])
                    emphasis=" is-selected" if species==label else ""
                    st.markdown(f'''<div class="score-row"><span class="score-label{emphasis}">{species}</span>
                    <div class="score-track" role="progressbar" aria-label="Skor {species}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{value*100:.1f}">
                    <div class="score-fill{emphasis}" style="width:{value*100:.3f}%"></div></div>
                    <span class="score-number">{value:.1%}</span></div>''',unsafe_allow_html=True)
                for warning in saved["warnings"]:st.warning(warning)
                st.info("Hasil merupakan perkiraan, bukan kepastian. Aplikasi hanya mengenali Adelie, Chinstrap, dan Gentoo.")
                download=pd.DataFrame([saved["input"]|record])
                st.download_button("Unduh hasil prediksi",download.to_csv(index=False).encode(),"hasil_penguin.csv","text/csv",width="stretch")
                if selected=="KNN":
                    with st.expander("Lihat dasar prediksi"):
                        row=pd.DataFrame([saved["input"]])[FEATURES]
                        transformed=model[:-1].transform(row)
                        distances,indices=model.named_steps["classifier"].kneighbors(transformed)
                        source=pd.read_csv(ROOT / "data" / "penguins.csv")
                        ids=read_json(ROOT / "results" / "split_indices.json")["train"]
                        neighbors=source.loc[ids].iloc[indices[0]][["species"]+FEATURES].copy()
                        neighbors["jarak_terstandardisasi"]=distances[0]
                        st.write("Lima catatan pembanding dengan ukuran tubuh paling mirip:")
                        column_names={c:f"{DISPLAY[c][0]} ({DISPLAY[c][1]})" for c in FEATURES}
                        column_names.update({"species":"Spesies","jarak_terstandardisasi":"Jarak kemiripan"})
                        st.dataframe(neighbors.rename(columns=column_names),hide_index=True,width="stretch")
                        st.caption("Semakin kecil jaraknya, semakin mirip ukuran tubuhnya. Perhitungan memakai ukuran yang telah disetarakan skalanya.")
            else:
                st.markdown(f'<div class="species">{penguin_svg()}<div><h2>Siap menebak</h2><p>Isi ukuran tubuh dan tekan Prediksi spesies.</p></div></div>',unsafe_allow_html=True)
                st.info("Belum ada prediksi. Masukkan ukuran tubuh atau pilih salah satu contoh untuk memulai.")

    with batch_tab:
        st.subheader("Prediksi beberapa penguin melalui CSV")
        st.write("File UTF-8, pemisah koma, titik untuk desimal; ukuran dalam mm dan gram. Kolom tambahan diabaikan model.")
        st.code(",".join(FEATURES))
        a,b=st.columns(2)
        a.download_button("Unduh template CSV",(ROOT / "examples" / "template_input.csv").read_bytes(),"template_input.csv","text/csv",width="stretch")
        b.download_button("Unduh contoh batch",(ROOT / "examples" / "contoh_batch.csv").read_bytes(),"contoh_batch.csv","text/csv",width="stretch")
        uploaded=st.file_uploader("Unggah CSV penguin",type=["csv"],key="batch_upload")
        if uploaded is not None:
            try:
                if uploaded.size > 5_000_000:raise ValueError("Ukuran file maksimal 5 MB.")
                content=uploaded.getvalue().decode("utf-8-sig")
                headers=next(csv.reader(io.StringIO(content)),[])
                if len(headers)!=len(set(headers)):raise ValueError("Nama kolom CSV tidak boleh duplikat.")
                frame=pd.read_csv(io.StringIO(content))
                st.dataframe(frame.head(10),hide_index=True,width="stretch")
                if st.button("Proses batch",type="primary"):
                    output,warnings=predict_frame(model,frame,summary)
                    for warning in warnings:st.warning(warning)
                    st.success(f"Selesai memprediksi {len(output)} penguin.")
                    st.dataframe(output,hide_index=True,width="stretch")
                    st.bar_chart(output.prediksi.value_counts())
                    st.download_button("Unduh hasil batch",output.to_csv(index=False).encode(),"hasil_batch_penguin.csv","text/csv",width="stretch")
            except (ValueError,UnicodeError,pd.errors.ParserError) as exc:
                st.error(f"CSV belum dapat diproses: {exc}")
            except Exception as exc:
                logger.exception("Gagal memproses unggahan CSV")
                st.error("File belum dapat diproses. Periksa format CSV atau coba kembali nanti.")


def show_evaluation():
    st.title("Tentang model")
    st.write("Kenali metode yang digunakan, hasil evaluasi, dan batas kemampuan prediksi aplikasi.")
    a,b,c,d=st.columns(4)
    a.metric("Model aktif",selected)
    b.metric("CV macro-F1",f"{selected_cv.cv_f1_macro_mean:.3f}",help="Rata-rata lima fold pada training; pemilihan utama.")
    c.metric("Test accuracy",f"{selected_test.accuracy:.1%}")
    d.metric("Data test",summary["n_test"])
    st.info("Hasil evaluasi yang tinggi pada data terbatas tidak menjamin semua pengukuran baru akan dikenali dengan benar. Skor prediksi belum diuji kalibrasinya.")
    cvtab,testtab,treetab,featuretab=st.tabs(["Validasi & pemilihan","Test akhir","Pohon Decision Tree","Fitur & kesalahan"])
    with cvtab:
        primary=cv[cv.feature_set=="4 fitur"].copy()
        st.dataframe(primary[["model","cv_f1_macro_mean","cv_f1_macro_std","cv_train_f1_mean","cv_gap"]],hide_index=True,width="stretch")
        st.image(str(ROOT / "plots" / "07_cv_models.png"),width="stretch")
        st.caption("SD memakai ddof=1; selisih CV KNN dan LR kecil, tidak menunjukkan keunggulan yang signifikan secara statistik.")
        with st.expander("Detail metode dan parameter"):
            st.write("Data dibagi 80:20 dengan proporsi spesies terjaga. Model dipilih melalui validasi lima fold pada training, sebelum test. Pengisian nilai kosong dan penyetaraan skala dipelajari hanya dari bagian latih masing-masing fold.")
            st.json(summary["parameters"])
        with st.expander("Perbandingan dua dan empat ukuran tubuh"):
            st.image(str(ROOT / "plots" / "10_feature_experiment.png"),width="stretch")
            st.dataframe(cv[["feature_set","model","cv_f1_macro_mean","cv_f1_macro_std"]],hide_index=True,width="stretch")
            st.caption("Analisis menggunakan validasi training. Prediksi aplikasi memakai empat ukuran tubuh.")
    with testtab:
        st.dataframe(test,hide_index=True,width="stretch")
        st.image(str(ROOT / "plots" / "06_confusion_matrix.png"),width="stretch")
        report=read_json(ROOT / "results" / "classification_reports.json")
        chosen=st.selectbox("Classification report",list(report),key="report_model")
        st.dataframe(pd.DataFrame(report[chosen]).T,width="stretch")
        st.caption("Baris confusion matrix = aktual; kolom = prediksi. Macro memberi bobot sama pada setiap spesies.")
        baseline=read_json(ROOT / "results" / "baseline.json")
        st.write(f"Baseline selalu kelas mayoritas: accuracy {baseline['accuracy']:.1%}, macro-F1 {baseline['f1_macro']:.3f}.")
    with treetab:
        st.subheader("Satu pohon, pertanyaan bercabang")
        st.write("Pohon ini merupakan kandidat Decision Tree, bukan otomatis model yang sedang dipakai untuk prediksi.")
        st.image(str(ROOT / "plots" / "09_decision_tree.png"),width="stretch")
        st.code((ROOT / "results" / "decision_tree_rules.txt").read_text(),language="text")
        st.caption("Ambang berasal dari training. Kedalaman dibatasi 3 dan setiap daun minimal 3 contoh. Angka pada node melibatkan data training setelah imputasi.")
    with featuretab:
        st.image(str(ROOT / "plots" / "08_feature_importance.png"),width="stretch")
        st.caption("Permutation importance dihitung pada validation training, bukan holdout test; bukan bukti sebab-akibat. Fitur berkorelasi dapat saling menggantikan.")
        st.subheader("Kesalahan test semua kandidat")
        errors=pd.read_csv(ROOT / "results" / "test_errors.csv")
        st.dataframe(errors,hide_index=True,width="stretch")
        st.subheader("Kesalahan out-of-fold model terpilih")
        oof=pd.read_csv(ROOT / "results" / "oof_predictions.csv")
        st.dataframe(oof[(oof.model==selected)&(~oof.correct)],hide_index=True,width="stretch")
        st.caption("Walaupun test terpilih bisa sempurna, OOF menunjukkan kesalahan yang benar-benar terjadi selama validasi.")


def show_dataset():
    st.title("Tentang dataset")
    st.write("Palmer Penguins: penguin dewasa dari Palmer Archipelago, Antarktika; dikumpulkan pada 2007–2009 oleh Kristen Gorman dan Palmer Station LTER.")
    st.link_button("Dokumentasi sumber","https://allisonhorst.github.io/palmerpenguins/")
    st.write("**Target:** species. **Input model:** empat ukuran tubuh. Pulau, sex, dan tahun dijelaskan, tetapi tidak dipakai sebagai prediktor utama.")
    st.dataframe(pd.DataFrame([{"fitur":key,"arti":DISPLAY[key][0],"satuan":DISPLAY[key][1]} for key in FEATURES]),hide_index=True,width="stretch")
    a,b,c=st.columns(3)
    a.metric("Jumlah penguin",summary["n_source"])
    b.metric("Data training",summary["n_train"])
    c.metric("Duplikat sumber",profile["duplicates_full"])
    st.json({"jumlah_per_spesies":profile["class_counts"],"nilai_kosong":profile["missing"]},expanded=False)
    st.caption("Missing dua pada masing-masing fitur tidak berarti delapan penguin berbeda. Sex memiliki missing, tetapi bukan input model.")
    captions={
        "01_distribusi_kelas.png":"Jumlah kelas training tidak sama; stratify dan macro-F1 dipakai agar tiap spesies diperhatikan.",
        "02_histogram.png":"Histogram menunjukkan rentang dan sebaran ukuran. Bandingkan kurva setiap spesies, bukan angka rata-rata saja.",
        "03_boxplot.png":"Boxplot per spesies mencegah ukuran besar satu spesies langsung dianggap kesalahan. Tidak ada penghapusan outlier otomatis.",
        "04_scatter.png":"Panjang dan kedalaman paruh membantu membedakan kelompok; beberapa wilayah tumpang tindih. Sirip/berat juga memberikan informasi ukuran tubuh.",
        "05_korelasi.png":"Korelasi adalah hubungan numerik, bukan sebab-akibat; pola gabungan spesies dapat berbeda dari pola dalam spesies.",
    }
    for filename,caption in captions.items():
        with st.expander(filename[3:-4].replace("_"," ").title()):
            st.image(str(ROOT / "plots" / filename),width="stretch")
            st.write(caption)
    with st.expander("Lihat data sumber"):
        frame=pd.read_csv(ROOT / "data" / "penguins.csv")
        st.dataframe(frame,hide_index=True,width="stretch")
        st.download_button("Unduh dataset sumber",(ROOT / "data" / "penguins.csv").read_bytes(),"penguins.csv","text/csv")
    st.warning("Keterbatasan: data kecil, tiga spesies dewasa, wilayah/periode terbatas, split acak bukan validasi lokasi baru. Model tidak mengukur dampak iklim atau menggantikan identifikasi ahli.")


def show_help():
    st.title("Cara menggunakan")
    st.write("Identifikasi spesies penguin melalui empat pengukuran tubuh atau unggahan CSV.")
    st.subheader("Prediksi satu penguin")
    st.markdown("1. Buka **Prediksi** dan pilih **Prediksi satu penguin**.\n2. Masukkan empat ukuran tubuh sesuai satuannya. Anda juga dapat memakai tombol contoh.\n3. Tekan **Prediksi spesies** untuk melihat hasil dan skor setiap spesies.\n4. Gunakan **Unduh hasil prediksi** untuk menyimpan hasil atau **Reset form** untuk memulai kembali.")
    st.subheader("Arti ukuran dan satuan")
    st.dataframe(pd.DataFrame([
        {"Ukuran":"Panjang paruh","Keterangan":"Panjang paruh dari pangkal ke ujung.","Satuan":"Milimeter (mm)"},
        {"Ukuran":"Kedalaman paruh","Keterangan":"Ketebalan vertikal paruh pada pangkal.","Satuan":"Milimeter (mm)"},
        {"Ukuran":"Panjang sirip","Keterangan":"Panjang sirip penguin.","Satuan":"Milimeter (mm)"},
        {"Ukuran":"Berat tubuh","Keterangan":"Berat seluruh tubuh. 3,75 kg = 3750 gram.","Satuan":"Gram"},
    ]),hide_index=True,width="stretch")
    st.subheader("Prediksi dari CSV")
    st.markdown("1. Buka tab **Batch CSV** pada halaman Prediksi.\n2. Unduh template dan isi satu penguin per baris. Gunakan titik untuk angka desimal.\n3. Unggah file, lalu tekan **Proses batch**.\n4. Periksa hasil dan unduh CSV hasil prediksi.")
    st.caption("Maksimal 5 MB dan 5.000 baris per file. Empat kolom ukuran harus tersedia. Nilai kosong parsial akan diisi otomatis dengan peringatan; baris yang seluruh ukurannya kosong tidak dapat diproses.")
    st.subheader("Memahami hasil")
    st.write("Nama spesies adalah perkiraan model. Skor menunjukkan keluaran model, bukan tingkat kepastian mutlak. Contoh pengukuran berasal dari dataset Palmer Penguins. Ilustrasi penguin bersifat dekoratif, bukan panduan membedakan anatomi spesies.")
    st.warning("Aplikasi hanya mendukung Adelie, Chinstrap, dan Gentoo. Ukuran di luar rentang data acuan atau satuan yang keliru dapat membuat hasil kurang andal.")


if nav=="Prediksi":show_single()
elif nav=="Tentang model":show_evaluation()
elif nav=="Tentang dataset":show_dataset()
else:show_help()
st.markdown('<div class="footer">Penguin Classifier · Identifikasi tiga spesies dari ukuran tubuh · Hasil prediksi bukan kepastian.</div>',unsafe_allow_html=True)
