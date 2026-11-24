# hr-attrition-prediction-fastapi-streamlit
Machine Learrning untuk memprediksi perkiraan karyawan yang akan resign dari suatu perusahaan. Dengan menggunakan bahasa pemrogram 
python yang diintegrasikan dengan Gemini AI. Data set yang digunakan diunduh dari 
Kaggle (https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).
Sebelum melakukan prediksi data, data tersebut dilakukan tranning dengan nama modul train.py terlebih dahulu agar 
tercipta file data model joblib.

Struktur Folder

hr-attrition-app/
├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Dataset CSV karyawan (1.470 baris)
├── train.py                                   # Pelatihan & evaluasi model ML (.joblib)
├── database.py                                # Inisialisasi SQLite, masking API Key, & log prediksi
├── schemas.py                                 # Skema validasi request & response (Pydantic)
├── main.py                                    # Server FastAPI, auth X-API-Key, & dashboard HTML
├── frontend_app.py                            # Frontend Python (Streamlit) + Gemini LLM
└── README.md                                  # Dokumentasi penggunaan aplikasi


