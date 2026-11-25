# HR Attrition Prediction FastAPI Streamlit
<p><strong>Aplikasi Prediksi Resign Karyawan &amp; Analytics Strategis HR</strong><span style="font-weight: 400;"> adalah solusi berbasis kecerdasan buatan (AI) dan Machine Learning yang dirancang untuk membantu Departemen Human Resources (HR) serta jajaran manajemen dalam mengidentifikasi risiko </span><em><span style="font-weight: 400;">attrition</span></em><span style="font-weight: 400;"> (pengunduran diri) karyawan secara dini.</span></p>
<p><span style="font-weight: 400;">Pada mini projek ini menggunakan model algoritma Random Forest Clasfier dengan pertimbangan berdasarkan pada kesesuaian karakteristik algoritma tersebut dengan struktur data HR dan kebutuhan operasional aplikasi:</span></p>
<ul>
<li style="font-weight: 400;"><strong>Menangani Fitur Campuran (Numerik &amp; Kategorikal) dengan Baik</strong></li>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Fitur input pada proyek ini terdiri dari variabel numerik (</span><strong>Age</strong><strong>, </strong><strong>MonthlyIncome</strong><strong>, </strong><strong>YearsAtCompany</strong><strong>, </strong><strong>EnvironmentSatisfaction</strong><span style="font-weight: 400;">) dan kategorikal (</span><strong>OverTime</strong><strong>, </strong><strong>MaritalStatus</strong><strong>, </strong><strong>JobRole</strong><span style="font-weight: 400;">).</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Berbasis pohon keputusan (</span><em><span style="font-weight: 400;">decision trees</span></em><span style="font-weight: 400;">), Random Forest mudah menangkap batas pemisah (</span><em><span style="font-weight: 400;">split</span></em><span style="font-weight: 400;">) non-linear pada fitur-fitur tersebut tanpa mengasumsikan hubungan linear seperti pada regresi logistik.</span></li>
</ul>
<li style="font-weight: 400;"><strong>Robust terhadap Pola Non-Linear &amp; Interaksi Fitur</strong></li>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Risiko </span><em><span style="font-weight: 400;">resign</span></em><span style="font-weight: 400;"> karyawan umumnya muncul dari kombinasi beberapa faktor sekaligus (misalnya karyawan muda dengan </span><strong>YearsAtCompany</strong><span style="font-weight: 400;"> singkat, </span><strong>MonthlyIncome</strong><span style="font-weight: 400;"> rendah, dan </span><strong>OverTime = Yes</strong><span style="font-weight: 400;">), yang ditangkap secara alami melalui percabangan pada banyak pohon keputusan.</span></li>
</ul>
<li style="font-weight: 400;"><strong>Mengurangi Risiko </strong><strong><em>Overfitting</em></strong><strong> Dibandingkan Single Decision Tree</strong></li>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Dengan menggabungkan banyak pohon yang dilatih pada </span><em><span style="font-weight: 400;">bootstrap sample</span></em><span style="font-weight: 400;"> dan subset fitur acak (</span><em><span style="font-weight: 400;">bagging</span></em><span style="font-weight: 400;"> + </span><em><span style="font-weight: 400;">feature subsampling</span></em><span style="font-weight: 400;">), varians prediksi menjadi lebih stabil dibanding satu pohon keputusan tunggal.</span></li>
</ul>
<li style="font-weight: 400;"><strong>Mendukung Penanganan Ketidakseimbangan Kelas (</strong><strong>class_weight</strong><strong>)</strong></li>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Pada dataset HR, jumlah karyawan yang tidak </span><em><span style="font-weight: 400;">resign</span></em><span style="font-weight: 400;"> (</span><span style="font-weight: 400;">No</span><span style="font-weight: 400;">: 1.233) jauh lebih banyak dibanding yang </span><em><span style="font-weight: 400;">resign</span></em><span style="font-weight: 400;"> (</span><strong>Yes</strong><span style="font-weight: 400;">: 237). Random Forest mendukung </span><span style="font-weight: 400;">class_weight</span><span style="font-weight: 400;"> (seperti </span><span style="font-weight: 400;">balanced_subsample</span><span style="font-weight: 400;">) agar kelas minoritas (</span><span style="font-weight: 400;">Yes</span><span style="font-weight: 400;">) tetap terwakili saat pelatihan.</span></li>
</ul>
<li style="font-weight: 400;"><strong>Menghasilkan Probabilitas Terkalibrasi &amp; Kompatibel dengan Pipeline Scikit-Learn</strong></li>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Metode </span><span style="font-weight: 400;">predict_proba()</span><span style="font-weight: 400;"> memudahkan penghitungan skor probabilitas </span><em><span style="font-weight: 400;">resign</span></em><span style="font-weight: 400;"> serta pengaturan </span><em><span style="font-weight: 400;">threshold</span></em><span style="font-weight: 400;"> keputusan di API/dashboard sesuai sensitivitas yang dibutuhkan tim HR.</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Mudah disatukan ke dalam </span><span style="font-weight: 400;">Pipeline</span><span style="font-weight: 400;"> scikit-learn bersama </span><span style="font-weight: 400;">ColumnTransformer</span><span style="font-weight: 400;"> (</span><span style="font-weight: 400;">SimpleImputer</span><span style="font-weight: 400;">, </span><span style="font-weight: 400;">StandardScaler</span><span style="font-weight: 400;">, </span><span style="font-weight: 400;">OneHotEncoder</span><span style="font-weight: 400;">) dan disimpan ke format </span><span style="font-weight: 400;">.joblib</span><span style="font-weight: 400;"> untuk inferensi cepat di FastAPI.</span></li>
</ul>
</ul>
<p><span style="font-weight: 400;">Aplikasi ini mengintegrasikan pemodelan prediktif berbasis data internal perusahaan dengan kapabilitas Large Language Model (LLM) dari </span><strong>Google Gemini</strong><span style="font-weight: 400;">, sehingga tidak hanya menyajikan angka probabilitas resign, tetapi juga memberikan analisis naratif dan rekomendasi aksi retensi yang konkret, kontekstual, dan berorientasi strategis.</span></p>

# Prerequisite (Prasyarat) Dan Installasi

<h2><strong>1. Struktur Folder</strong></h2>
<img width="768" height="143" alt="image" src="https://github.com/user-attachments/assets/9e9e74a5-0d57-4964-beec-2eb0906ac765" />
<p></p>
<h2><strong>2. Langkah-Langkah Instalasi</strong></h2>
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

<img width="803" height="170" alt="image" src="https://github.com/user-attachments/assets/5e64aa48-b098-49c5-9ef4-7861edf86e6a" />


<p><span style="font-weight: 400;">Jika belum memiliki </span><span style="font-weight: 400;">requirements.txt</span><span style="font-weight: 400;">, Anda dapat menginstalnya langsung:</span></p>

<img width="815" height="172" alt="image" src="https://github.com/user-attachments/assets/4b9810cb-6225-451f-8124-c6e9835b1554" />


<h4><strong>Langkah 4: Verifikasi Dataset &amp; Konfigurasi Awal</strong></h4>
<ol>
<li style="font-weight: 400;"><span style="font-weight: 400;">Pastikan berkas dataset CSV berada di folder proyek dan nama berkasnya (</span><strong>DATA_PATH</strong><span style="font-weight: 400;">) cocok dengan yang direferensikan di </span><strong>train.py</strong><span style="font-weight: 400;"> dan </span><strong>main.py</strong><span style="font-weight: 400;">.</span></li>
<li style="font-weight: 400;"><em><span style="font-weight: 400;">(Opsional)</span></em><span style="font-weight: 400;"> Siapkan variabel lingkungan sebelum menjalankan aplikasi:</span></li>
</ol>
<p><strong><em>Windows (PowerShell)</em></strong><strong> :</strong></p>
<p> </p>

<img width="740" height="204" alt="image" src="https://github.com/user-attachments/assets/c7ee3781-b1e1-4a10-913a-fed2bef4af08" />


<p> </p>
<p><strong><em>Linux / macOS</em></strong><strong> :</strong></p>
<img width="736" height="197" alt="image" src="https://github.com/user-attachments/assets/1c8a8ec3-0b0b-4ac3-b07f-0bc82a98dc73" />

<p> </p>
<h4><strong>Langkah 5: Latih Model Machine Learning</strong></h4>
<p><span style="font-weight: 400;">Jalankan </span><strong>train.py</strong><span style="font-weight: 400;"> untuk melatih model </span><strong>RandomForestClassifier</strong><span style="font-weight: 400;"> (7 fitur input) dan menghasilkan berkas model </span><span style="font-weight: 400;">.</span><strong>joblib </strong><strong>:</strong></p>
<img width="815" height="166" alt="image" src="https://github.com/user-attachments/assets/25b7a068-1091-4ae7-b566-524d1b0a38be" />

<h4><strong>Langkah 6: Jalankan Backend FastAPI &amp; Frontend Streamlit</strong></h4>
<p><strong>Terminal 1 (Backend FastAPI):</strong></p>
<img width="789" height="157" alt="image" src="https://github.com/user-attachments/assets/53280b2a-7448-47b3-af60-a4887d715a06" />
<p> </p>
<img width="773" height="311" alt="image" src="https://github.com/user-attachments/assets/7c67f80d-9aea-40da-994c-1f276842b994" />

<p> </p>
<p><strong>Terminal 2 (Frontend Python Streamlit — dengan </strong><strong>.venv</strong><strong> aktif):</strong></p>
<p></p>
<img width="810" height="165" alt="image" src="https://github.com/user-attachments/assets/bd7607f0-f995-4d86-94fd-d58ddcdead4f" />
<p> </p>
<img width="570" height="331" alt="image" src="https://github.com/user-attachments/assets/7d2cec2c-db5f-4d71-bdfe-03b4fdba773d" />

<p></p>
<p></p>
<p>Setelah Terminal 1 dan Terminal 2 berjalan dengan baik maka selanjutnya adalah membuka aplikasi browser untuk mengakses aplikasi tersebut</p>
<p></p>
<h4><strong>Langkah 7 : Mengakses FastAPI</strong></h4>
<ol style="list-style-type: upper-alpha;">
<li><strong><span style="font-weight: 400;">Base URL API ==&gt; </span><span style="font-weight: 400;">http://127.0.0.1:8000</span></strong></li>
<li><span style="font-weight: 400;">Dokumentasi Interaktif (Swagger) ==&gt; </span><span style="font-weight: 400;">http://127.0.0.1:8000/docs</span></li>
<li><span style="font-weight: 400;">Dashboard HTML/Chart.js ==&gt; </span><span style="font-weight: 400;">http://127.0.0.1:8000/dashboard</span></li>
</ol>
<p> </p>
<h4><strong>Langkah 8 : Menjalankan Frontend Python (Streamlit)</strong></h4>

<p><strong> A. Konfigurasi</strong></p>

<ol>
<li style="font-weight: 400;"><span style="font-weight: 400;">Isi </span><strong>Base URL FastAPI</strong><span style="font-weight: 400;"> dengan </span><a href="http://127.0.0.1:8000"><span style="font-weight: 400;">http://127.0.0.1:8000</span></a></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Isi </span><strong>X-API-Key</strong><span style="font-weight: 400;"> (default: </span><span style="font-weight: 400;">dev-secret-key-123</span><span style="font-weight: 400;">)</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Atur </span><strong>Threshold Prediksi Resign</strong><span style="font-weight: 400;"> (default: </span><span style="font-weight: 400;">0.40</span><span style="font-weight: 400;">).</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Isi </span><strong>Gemini API Key</strong><span style="font-weight: 400;"> (wajib untuk fitur analisa LLM) </span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;"> pilih model Gemini ( gemini-3.5-flash,</span><span style="font-weight: 400;">gemini-2.5-flash</span><span style="font-weight: 400;"> atau </span><span style="font-weight: 400;">gemini-2.5-pro</span><span style="font-weight: 400;">).</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Klik </span><strong>Test Koneksi Gemini</strong><span style="font-weight: 400;"> untuk memverifikasi key dan mengecek status koneksi/error sebelum melakukan prediksi.</span></li>
</ol>
<h4><strong>B. Prediksi Resign &amp; Analisa Gemini</strong></h4>
<ol>
<li style="font-weight: 400;"><span style="font-weight: 400;">Isi 7 parameter input:</span></li>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">OverTime</span><span style="font-weight: 400;"> (</span><span style="font-weight: 400;">Yes</span><span style="font-weight: 400;"> / </span><span style="font-weight: 400;">No</span><span style="font-weight: 400;">)</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">MonthlyIncome</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Age</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">MaritalStatus</span><span style="font-weight: 400;"> (</span><span style="font-weight: 400;">Single</span><span style="font-weight: 400;"> / </span><span style="font-weight: 400;">Married</span><span style="font-weight: 400;"> / </span><span style="font-weight: 400;">Divorced</span><span style="font-weight: 400;">)</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">YearsAtCompany</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">EnvironmentSatisfaction</span><span style="font-weight: 400;"> (</span><span style="font-weight: 400;">1</span><span style="font-weight: 400;">–</span><span style="font-weight: 400;">4</span><span style="font-weight: 400;">)</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">JobRole</span></li>
</ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Isi </span><strong>Ref Karyawan</strong><span style="font-weight: 400;"> (opsional) lalu klik </span><strong>Prediksi &amp; Analisa Gemini</strong><span style="font-weight: 400;">.</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Sistem menampilkan hasil prediksi (</span><span style="font-weight: 400;">Yes</span><span style="font-weight: 400;">/</span><span style="font-weight: 400;">No</span><span style="font-weight: 400;">), probabilitas </span><em><span style="font-weight: 400;">resign</span></em><span style="font-weight: 400;">, </span><span style="font-weight: 400;">Record ID</span><span style="font-weight: 400;"> tersimpan di SQLite, </span><em><span style="font-weight: 400;">masked API Key</span></em><span style="font-weight: 400;">, serta ringkasan faktor risiko dan rekomendasi retensi dari Gemini.</span></li>
</ol>
 
