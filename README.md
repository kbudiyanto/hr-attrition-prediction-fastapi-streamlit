# HR Attrition Prediction FastAPI Streamlit
<p><strong>Aplikasi Prediksi Resign Karyawan &amp; Analytics Strategis HR</strong><span style="font-weight: 400;"> adalah solusi berbasis kecerdasan buatan (AI) dan Machine Learning yang dirancang untuk membantu Departemen Human Resources (HR) serta jajaran manajemen dalam mengidentifikasi risiko </span><em><span style="font-weight: 400;">attrition</span></em><span style="font-weight: 400;"> (pengunduran diri) karyawan secara dini.</span></p>
<p><span style="font-weight: 400;">Aplikasi ini mengintegrasikan pemodelan prediktif berbasis data internal perusahaan dengan kapabilitas Large Language Model (LLM) dari </span><strong>Google Gemini</strong><span style="font-weight: 400;">, sehingga tidak hanya menyajikan angka probabilitas resign, tetapi juga memberikan analisis naratif dan rekomendasi aksi retensi yang konkret, kontekstual, dan berorientasi strategis.</span></p>
<h3><strong>Arsitektur &amp; Pilar Utama Sistem</strong></h3>
<p><span style="font-weight: 400;">Sistem ini dibangun dengan arsitektur modular yang memisahkan peran </span><em><span style="font-weight: 400;">machine learning core</span></em><span style="font-weight: 400;">, </span><em><span style="font-weight: 400;">backend API</span></em><span style="font-weight: 400;">, </span><em><span style="font-weight: 400;">layanan persistence</span></em><span style="font-weight: 400;">, serta </span><em><span style="font-weight: 400;">frontend antarmuka pengembang dan pengambil keputusan</span></em><span style="font-weight: 400;">:</span></p>
<ol>
<li style="font-weight: 400;"><strong>Mesin Prediksi Machine Learning (Scikit-Learn)</strong></li>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Ditingkatkan menggunakan algoritma </span><strong>Random Forest Classifier</strong><span style="font-weight: 400;"> yang dilatih di atas dataset indikator HR (seperti </span><em><span style="font-weight: 400;">Age</span></em><span style="font-weight: 400;">, </span><em><span style="font-weight: 400;">JobRole</span></em><span style="font-weight: 400;">, </span><em><span style="font-weight: 400;">MaritalStatus</span></em><span style="font-weight: 400;">, </span><em><span style="font-weight: 400;">MonthlyIncome</span></em><span style="font-weight: 400;">, </span><em><span style="font-weight: 400;">YearsAtCompany</span></em><span style="font-weight: 400;">, </span><em><span style="font-weight: 400;">EnvironmentSatisfaction</span></em><span style="font-weight: 400;">, dan </span><em><span style="font-weight: 400;">OverTime</span></em><span style="font-weight: 400;">).</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Dilengkapi validasi silang (</span><em><span style="font-weight: 400;">5-Fold Stratified Cross-Validation</span></em><span style="font-weight: 400;">) untuk memastikan akurasi dan keandalan model saat menilai tingkat risiko karyawan.</span></li>
</ul>
<li style="font-weight: 400;"><strong>Backend Berkinerja Tinggi &amp; Keamanan (FastAPI &amp; SQLite)</strong></li>
<ul>
<li style="font-weight: 400;"><strong>FastAPI</strong><span style="font-weight: 400;"> bertindak sebagai fondasi REST API yang cepat, teringan, dan terstruktur untuk memproses permintaan prediksi secara </span><em><span style="font-weight: 400;">real-time</span></em><span style="font-weight: 400;">.</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Keamanan akses dijamin melalui mekanisme otentikasi </span><strong>API Key (X-API-Key)</strong><span style="font-weight: 400;">.</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Database </span><strong>SQLite</strong><span style="font-weight: 400;"> terintegrasi menyimpan seluruh riwayat transaksi prediksi dan secara otomatis menerapkan pengamanan </span><em><span style="font-weight: 400;">masking</span></em><span style="font-weight: 400;"> API Key untuk menjaga privasi data sensitif di lingkungan log.</span></li>
</ul>
<li style="font-weight: 400;"><strong>Antarmuka Pengguna Interaktif (Streamlit &amp; Dashboard HTML)</strong></li>
<ul>
<li style="font-weight: 400;"><strong>Dashboard HTML (Chart.js):</strong><span style="font-weight: 400;"> Menyajikan visualisasi real-time berbasis web untuk pemantauan cepat riwayat prediksi dan KPI karyawan.</span></li>
<li style="font-weight: 400;"><strong>Frontend Streamlit (Python):</strong><span style="font-weight: 400;"> Menyediakan formulir input interaktif 7 parameter, visualisasi grafik distribusi per posisi (</span><em><span style="font-weight: 400;">JobRole</span></em><span style="font-weight: 400;">), tabel matriks komparatif antara data historis dan hasil prediksi, serta alat pengujian koneksi sistem.</span></li>
</ul>
<li style="font-weight: 400;"><strong>Kecerdasan Buatan Generatif (Gemini LLM Integration)</strong></li>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Menggunakan SDK resmi google-genai (</span><em><span style="font-weight: 400;">Gemini 2.5 Flash</span></em><span style="font-weight: 400;"> dan </span><em><span style="font-weight: 400;">Gemini 2.5 Pro</span></em><span style="font-weight: 400;">) untuk menerjemahkan angka-angka prediksi menjadi narasi saran HR.</span></li>
<li style="font-weight: 400;"><strong>Analisis Individu:</strong><span style="font-weight: 400;"> Menghasilkan rekomendasi retensi spesifik untuk karyawan tertentu berdasarkan profil risiko individualnya.</span></li>
<li style="font-weight: 400;"><strong>Analisis Strategis Lintas Posisi:</strong><span style="font-weight: 400;"> Menganalisis matriks risiko secara agregat untuk mengidentifikasi divisi/posisi yang paling rawan </span><em><span style="font-weight: 400;">turnover</span></em><span style="font-weight: 400;"> serta memberikan panduan kebijakan retensi berskala organisasi.</span></li>
</ul>
</ol>

# Prerequisite (Prasyarat) Dan Installasi
<p><span style="text-decoration: underline;"><strong>Struktur Folder</strong></span></p>
<img width="768" height="143" alt="image" src="https://github.com/user-attachments/assets/9e9e74a5-0d57-4964-beec-2eb0906ac765" />
<p></p>
<h3><strong>2. Langkah-Langkah Instalasi</strong></h3>
<h4><strong>Langkah 1: Masuk ke Folder Proyek &amp; Buat Virtual Environment</strong></h4>
<p><span style="font-weight: 400;">Buka terminal di dalam folder proyek (misalnya </span><span style="font-weight: 400;">hr-attrition-app</span><span style="font-weight: 400;">), lalu buat </span><em><span style="font-weight: 400;">virtual environment</span></em><span style="font-weight: 400;">:</span></p>

<img width="824" height="154" alt="image" src="https://github.com/user-attachments/assets/1878e660-0985-4ede-afe5-30f874bc1621" />

<h4><strong>Langkah 2: Aktifkan Virtual Environment</strong></h4>
<p><span style="font-weight: 400;">Aktifkan </span><strong>.venv</strong> <span style="font-weight: 400;">sesuai sistem operasi yang digunakan:</span></p>
<p><strong><em>Windows (PowerShell)</em></strong><strong> :</strong></p>
<p> </p>

<img width="777" height="149" alt="image" src="https://github.com/user-attachments/assets/97dfd96b-895f-4dac-b4eb-7107ff326164" />

<p> </p>
<p><strong><em>Linux / macOS</em></strong><strong> :</strong></p>
<p><span style="font-weight: 400;">Bash</span></p>

<img width="774" height="147" alt="image" src="https://github.com/user-attachments/assets/98742dbe-4623-4c16-aee1-cd53dd412e55" />


<p> </p>
<p><span style="font-weight: 400;">Pastikan tanda </span><strong>(.venv)</strong><span style="font-weight: 400;"> muncul di awal baris terminal sebelum melanjutkan.</span></p>
<h4><strong>Langkah 3: Upgrade pip &amp; Instal Seluruh Dependensi</strong></h4>
<p><span style="font-weight: 400;">Instal semua paket yang dibutuhkan melalui </span><span style="font-weight: 400;">requirements.txt</span><span style="font-weight: 400;">:</span></p>
<p><span style="font-weight: 400;">Bash</span></p>
<p><strong>python -m pip install --upgrade pip</strong></p>
<p><strong>pip install -r requirements.txt</strong></p>
<p><span style="font-weight: 400;">Jika belum memiliki </span><span style="font-weight: 400;">requirements.txt</span><span style="font-weight: 400;">, Anda dapat menginstalnya langsung:</span></p>
<p><span style="font-weight: 400;">Bash</span></p>
<p><strong>pip install fastapi uvicorn pydantic scikit-learn joblib pandas numpy requests streamlit google-genai</strong></p>
<h4><strong>Langkah 4: Verifikasi Dataset &amp; Konfigurasi Awal</strong></h4>
<ol>
<li style="font-weight: 400;"><span style="font-weight: 400;">Pastikan berkas dataset CSV berada di folder proyek dan nama berkasnya (</span><strong>DATA_PATH</strong><span style="font-weight: 400;">) cocok dengan yang direferensikan di </span><strong>train.py</strong><span style="font-weight: 400;"> dan </span><strong>main.py</strong><span style="font-weight: 400;">.</span></li>
<li style="font-weight: 400;"><em><span style="font-weight: 400;">(Opsional)</span></em><span style="font-weight: 400;"> Siapkan variabel lingkungan sebelum menjalankan aplikasi:</span></li>
</ol>
<p><strong><em>Windows (PowerShell)</em></strong><strong> :</strong></p>
<p> </p>
<p><strong>$env:API_KEY="dev-secret-key-123"</strong></p>
<p><strong>$env:PREDICT_THRESHOLD="0.40"</strong></p>
<p><strong>$env:GEMINI_API_KEY="AIza..."</strong></p>
<p> </p>
<p><strong><em>Linux / macOS</em></strong><strong> :</strong></p>
<p><span style="font-weight: 400;">Bash</span></p>
<p><strong>export API_KEY="dev-secret-key-123"</strong></p>
<p><strong>export PREDICT_THRESHOLD="0.40"</strong></p>
<p><strong>export GEMINI_API_KEY="AIza..."</strong></p>
<p> </p>
<h4><strong>Langkah 5: Latih Model Machine Learning</strong></h4>
<p><span style="font-weight: 400;">Jalankan </span><strong>train.py</strong><span style="font-weight: 400;"> untuk melatih model </span><strong>RandomForestClassifier</strong><span style="font-weight: 400;"> (7 fitur input) dan menghasilkan berkas model </span><span style="font-weight: 400;">.</span><strong>joblib </strong><strong>:</strong></p>
<p><span style="font-weight: 400;">Bash</span></p>
<p><strong>python train.py</strong></p>
<h4><strong>Langkah 6: Jalankan Backend FastAPI &amp; Frontend Streamlit</strong></h4>
<p><strong>Terminal 1 (Backend FastAPI):</strong></p>
<p><span style="font-weight: 400;">Bash</span></p>
<p><strong>uvicorn main:app --reload</strong></p>
<p> </p>
<p><strong>Terminal 2 (Frontend Python Streamlit — dengan </strong><strong>.venv</strong><strong> aktif):</strong></p>
<p><span style="font-weight: 400;">Bash</span></p>
<p><strong>streamlit run frontend_app.py</strong></p>
