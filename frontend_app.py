import os
from google import genai
from google.genai import errors as genai_errors
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Prediksi Resign Karyawan + Gemini LLM",
    page_icon="📊",
    layout="wide",
)

st.title("Prediksi Resign Karyawan & Rekomendasi Gemini LLM")
st.caption(
    "Frontend Python (Streamlit) · 7 Fitur Input · Matrix & Grafik per JobRole · Validasi & Error Handling Gemini"
)

JOB_ROLES = [
    "Sales Representative",
    "Laboratory Technician",
    "Human Resources",
    "Sales Executive",
    "Research Scientist",
    "Manufacturing Director",
    "Healthcare Representative",
    "Manager",
    "Research Director",
]




def validate_fastapi_key(base_url: str, api_key_val: str) -> tuple[bool, str]:
    """Memastikan X-API-Key tidak kosong dan tervalidasi oleh endpoint backend FastAPI."""
    if not is_non_empty(api_key_val):
        return (
            False,
            "X-API-Key tidak boleh kosong! Silakan isi **X-API-Key** pada sidebar kiri.",
        )

    clean_key = api_key_val.strip()
    try:
        resp = requests.get(
            f"{base_url}/api/jobrole-matrix",
            headers={"X-API-Key": clean_key},
            timeout=10,
        )
        if resp.status_code == 200:
            return True, "X-API-Key valid dan terhubung ke backend FastAPI."
        if resp.status_code in (401, 403):
            return (
                False,
                f"X-API-Key tidak valid ({resp.status_code}): Periksa kembali nilai **X-API-Key** Anda.",
            )
        return (
            False,
            f"Validasi X-API-Key gagal (HTTP {resp.status_code}): {resp.text}",
        )
    except requests.RequestException as exc:
        return (
            False,
            f"Tidak dapat memvalidasi X-API-Key karena server FastAPI di '{base_url}' tidak terjangkau: {exc}",
        )


def is_non_empty(value: str | None) -> bool:
    return bool(value and value.strip())


def format_gemini_error(exc: Exception) -> str:
    if isinstance(exc, genai_errors.APIError):
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        msg = getattr(exc, "message", None) or str(exc)

        if status in (400, "INVALID_ARGUMENT"):
            return f"Permintaan ke Gemini tidak valid ({status}): {msg}"
        if status in (401, 403, "UNAUTHENTICATED", "PERMISSION_DENIED"):
            return (
                f"Autentikasi Gemini gagal ({status}): Periksa kembali **Gemini API Key** Anda "
                f"atau izin akses model. Detail: {msg}"
            )
        if status in (404, "NOT_FOUND"):
            return f"Model Gemini tidak ditemukan ({status}): {msg}"
        if status in (429, "RESOURCE_EXHAUSTED"):
            return (
                f"Kuota / rate limit Gemini API terlampaui ({status}): Tunggu beberapa saat "
                f"lalu coba lagi. Detail: {msg}"
            )
        if status in (500, 502, 503, 504, "INTERNAL", "UNAVAILABLE"):
            return f"Layanan Gemini sedang bermasalah ({status}): Silakan coba lagi. Detail: {msg}"

        return f"Gemini APIError ({status or 'unknown'}): {msg}"

    if isinstance(exc, TimeoutError):
        return "Permintaan ke Gemini API kehabisan waktu (timeout). Silakan coba lagi."
    if isinstance(exc, ConnectionError):
        return f"Gagal terhubung ke jaringan / Gemini API: {exc}"

    return f"Terjadi error saat memanggil Gemini: {exc}"


def call_gemini(prompt: str, api_key_val: str, model_name: str) -> tuple[bool, str]:
    if not is_non_empty(api_key_val):
        return (
            False,
            "Gemini API Key tidak boleh kosong! Silakan isi **Gemini API Key** pada sidebar terlebih dahulu.",
        )

    try:
        client = genai.Client(api_key=api_key_val.strip())
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
    except Exception as exc:
        return False, format_gemini_error(exc)

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        return False, "Gemini tidak mengembalikan teks rekomendasi (respons kosong atau terfilter)."

    return True, text


# --- Sidebar Konfigurasi ---
with st.sidebar:
    st.header("Konfigurasi Backend FastAPI")
    api_url = st.text_input("Base URL FastAPI", value="http://127.0.0.1:8000").rstrip("/")
    
    api_key = st.text_input("X-API-Key", value="dev-secret-key-123", type="password")

    apikey=False
 
    threshold = st.slider(
        "Threshold Prediksi Resign",
        min_value=0.05,
        max_value=0.95,
        value=0.40,
        step=0.05,
    )
    
               
    st.divider()
    st.header("Konfigurasi Gemini LLM")
    gemini_api_key = st.text_input(
        "Gemini API Key (Wajib)",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Wajib diisi untuk menjalankan analisa & rekomendasi Gemini.",
    )   
    # if not is_non_empty(gemini_api_key):
    #     st.warning("Gemini API Key masih kosong.")
    # else:
    #     st.success("Gemini API Key terisi.")

    gemini_model = st.selectbox(
        "Model Gemini",
        options=["gemini-3.5-flash","gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
    )

    geminisukses=False
    if st.button("Connect....", use_container_width=True):
        ok, test_msg = call_gemini(
            "Balas singkat dengan kata: OK",
            gemini_api_key,
            gemini_model,
        )
        ok_key, msg_key = validate_fastapi_key(api_url, api_key)
        st.session_state["fastapi_key_status"] = (ok_key, msg_key)
        
        if ok:
            #st.success(f"Koneksi Gemini berhasil: {test_msg}")
            geminisukses=True
        else:
            st.error(test_msg)
            
        if not is_non_empty(gemini_api_key):
            st.warning("Gemini API Key masih kosong.")
        # else:
        #     st.success("Gemini API Key terisi.")     
                
        if not is_non_empty(api_key):
                st.warning("X-API-Key masih kosong.")
        # else:
        #     st.info("X-API-Key sudah diisi (belum diverifikasi ke server).")

        if "fastapi_key_status" in st.session_state:
            ok_key, msg_key = st.session_state["fastapi_key_status"]
            if ok_key:
                #st.success(msg_key)
                apikey=True
            else:
                st.error(msg_key)  
            
    st.divider()
    if geminisukses==True and apikey==True:
        if st.button("Refresh Matrix & Grafik", use_container_width=True):
            st.rerun()

if geminisukses==True and apikey==True:
# --- Form 7 Parameter Input ---
    st.subheader("Form Prediksi Potensi Resign")
    with st.form("predict_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            over_time = st.selectbox("OverTime", options=["Yes", "No"])
            monthly_income = st.number_input("MonthlyIncome", min_value=1, value=3200, step=100)
        with c2:
            age = st.number_input("Age", min_value=18, max_value=70, value=29, step=1)
            marital_status = st.selectbox("MaritalStatus", options=["Single", "Married", "Divorced"])
        with c3:
            years_at_company = st.number_input("YearsAtCompany", min_value=0, value=1, step=1)
            env_satisfaction = st.slider(
                "EnvironmentSatisfaction (1-4)", min_value=1, max_value=4, value=2
            )
        with c4:
            job_role = st.selectbox("JobRole", options=JOB_ROLES)
            employee_id_ref = st.text_input("Ref Karyawan (Opsional)", value="EMP-PY-01")

        submitted = st.form_submit_button("Prediksi & Analisa Gemini", type="primary")


    if submitted:
        if not is_non_empty(api_key):
            st.error("X-API-Key FastAPI tidak boleh kosong. Isi pada sidebar kiri.")
            st.stop()

        payload = {
            "employee_id_ref": employee_id_ref.strip() or None,
            "OverTime": over_time,
            "MonthlyIncome": int(monthly_income),
            "Age": int(age),
            "MaritalStatus": marital_status,
            "YearsAtCompany": int(years_at_company),
            "EnvironmentSatisfaction": int(env_satisfaction),
            "JobRole": job_role,
        }

        try:
            resp = requests.post(
                f"{api_url}/predict",
                params={"threshold": threshold},
                json=payload,
                headers={"X-API-Key": api_key.strip(), "Content-Type": "application/json"},
                timeout=15,
            )
            if resp.status_code == 200:
                result = resp.json()
                pred_label = result.get("attrition_prediction", "-")
                prob_val = float(result.get("resign_probability", 0.0))
                rec_id = result.get("record_id", "-")
                masked_key = result.get("api_key_masked", "-")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Hasil Prediksi", pred_label)
                m2.metric("Probabilitas Resign", f"{prob_val * 100:.2f}%")
                m3.metric("Record ID (SQLite)", str(rec_id))
                m4.metric("API Key", masked_key)

                emp_prompt = (
                    "Anda adalah konsultan HR Analytics. Analisis profil karyawan dan hasil prediksi "
                    "resign berikut dalam bahasa Indonesia yang ringkas, terstruktur, dan actionable:\n"
                    f"- Ref Karyawan: {employee_id_ref.strip() or '-'}\n"
                    f"- JobRole: {job_role}\n"
                    f"- Age: {age}\n"
                    f"- MaritalStatus: {marital_status}\n"
                    f"- MonthlyIncome: {monthly_income}\n"
                    f"- YearsAtCompany: {years_at_company}\n"
                    f"- EnvironmentSatisfaction (1-4): {env_satisfaction}\n"
                    f"- OverTime: {over_time}\n"
                    f"- Prediksi Resign: {pred_label} (Probabilitas: {prob_val * 100:.2f}%, Threshold: {threshold})\n\n"
                    "Sajikan:\n"
                    "1. Ringkasan faktor risiko utama.\n"
                    "2. 3-4 langkah rekomendasi retensi konkret untuk manajer/HR."
                )

                with st.spinner("Menyusun analisa & rekomendasi retensi dengan Gemini..."):
                    ok, advice_or_err = call_gemini(emp_prompt, gemini_api_key, gemini_model)

                if ok:
                    st.session_state["emp_gemini_advice"] = advice_or_err
                    st.session_state.pop("emp_gemini_error", None)
                else:
                    st.session_state.pop("emp_gemini_advice", None)
                    st.session_state["emp_gemini_error"] = advice_or_err
            else:
                st.error(f"Gagal memprediksi ({resp.status_code}): {resp.text}")
        except requests.RequestException as exc:
            st.error(f"Tidak dapat terhubung ke server FastAPI di '{api_url}': {exc}")


    if "emp_gemini_error" in st.session_state:
        st.markdown("### Analisa & Rekomendasi Gemini untuk Karyawan Ini")
        st.error(st.session_state["emp_gemini_error"])
    elif "emp_gemini_advice" in st.session_state:
        st.markdown("### Analisa & Rekomendasi Gemini untuk Karyawan Ini")
        st.info(st.session_state["emp_gemini_advice"])

    st.divider()

    # --- Matrix & Grafik per JobRole ---
    st.subheader("Matrix & Grafik Hasil per JobRole")
    matrix_df = None

    if not is_non_empty(api_key):
        st.warning("Isi **X-API-Key** di sidebar kiri untuk memuat matrix dan grafik per JobRole.")
    else:
        try:
            matrix_resp = requests.get(
                f"{api_url}/api/jobrole-matrix",
                headers={"X-API-Key": api_key.strip()},
                timeout=15,
            )
            if matrix_resp.status_code == 200:
                matrix_data = matrix_resp.json().get("matrix", [])
                if matrix_data:
                    matrix_df = pd.DataFrame(matrix_data)

                    g1, g2 = st.columns(2)
                    with g1:
                        st.markdown("**% Resign Aktual vs Rata-rata Prob. Prediksi (%) per JobRole**")
                        chart_rate_df = matrix_df.set_index("JobRole")[
                            ["actual_resign_rate_pct", "avg_pred_prob_pct"]
                        ]
                        st.bar_chart(chart_rate_df)

                    with g2:
                        st.markdown("**Komposisi Data Aktual: Resign (Yes) vs Bertahan (No) per JobRole**")
                        chart_count_df = matrix_df.set_index("JobRole")[["actual_yes", "actual_no"]]
                        st.bar_chart(chart_count_df)

                    st.markdown("**Tabel Matrix Laporan per JobRole**")
                    display_df = matrix_df.rename(
                        columns={
                            "JobRole": "JobRole",
                            "actual_yes": "Data Resign (Yes)",
                            "actual_no": "Data Bertahan (No)",
                            "actual_total": "Total Data",
                            "actual_resign_rate_pct": "% Resign Aktual",
                            "pred_yes": "Prediksi Yes (DB)",
                            "pred_no": "Prediksi No (DB)",
                            "avg_pred_prob_pct": "Rata-rata Prob. Prediksi (%)",
                        }
                    )
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                    st.markdown("### Analisa Rekomendasi Lintas JobRole (Gemini LLM)")
                    if st.button("Buat Rekomendasi Strategis per JobRole dengan Gemini"):
                        matrix_csv = matrix_df.to_csv(index=False)
                        role_prompt = (
                            "Anda adalah pakar People Analytics & HR Strategy. Analisis tabel matrix "
                            "resign dan prediksi per JobRole berikut dalam bahasa Indonesia:\n\n"
                            f"{matrix_csv}\n\n"
                            "Sajikan:\n"
                            "1. Posisi (JobRole) dengan prioritas risiko resign tertinggi dan polanya.\n"
                            "2. Rekomendasi kebijakan retensi terarah per kelompok peran prioritas."
                        )
                        with st.spinner("Menganalisis matrix per JobRole dengan Gemini..."):
                            ok, role_res = call_gemini(role_prompt, gemini_api_key, gemini_model)

                        if ok:
                            st.session_state["role_gemini_advice"] = role_res
                            st.session_state.pop("role_gemini_error", None)
                        else:
                            st.session_state.pop("role_gemini_advice", None)
                            st.session_state["role_gemini_error"] = role_res

                    if "role_gemini_error" in st.session_state:
                        st.error(st.session_state["role_gemini_error"])
                    elif "role_gemini_advice" in st.session_state:
                        st.success(st.session_state["role_gemini_advice"])
                else:
                    st.info("Data matrix per JobRole belum tersedia.")
            else:
                st.warning(f"Gagal memuat matrix ({matrix_resp.status_code}): {matrix_resp.text}")
        except requests.RequestException as exc:
            st.info(f"Menunggu koneksi ke FastAPI di '{api_url}' ({exc}).")
            
            
    #--------------