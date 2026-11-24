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
