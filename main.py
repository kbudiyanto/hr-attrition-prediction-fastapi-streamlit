from contextlib import asynccontextmanager
import os
from pathlib import Path
import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader

from database import (
    get_prediction_stats,
    get_predictions,
    init_db,
    mask_api_key,
    save_prediction,
)
from schemas import EmployeeInput, PredictionLogItem, PredictionResponse
from train import DATA_PATH, FEATURE_COLS, MODEL_PATH, train_and_save

EXPECTED_API_KEY = os.getenv("API_KEY", "123")
DEFAULT_THRESHOLD = float(os.getenv("PREDICT_THRESHOLD", "0.40"))
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
model_state: dict = {}


def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    if not api_key or api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="X-API-Key tidak valid atau kosong.")
    return api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    m_path = Path(os.getenv("MODEL_PATH", str(MODEL_PATH)))
    d_path = Path(os.getenv("DATA_PATH", str(DATA_PATH)))
    if not m_path.exists() and d_path.exists():
        train_and_save(data_path=d_path, model_path=m_path)
    if m_path.exists():
        model_state.update(joblib.load(m_path))
    yield


app = FastAPI(title="HR Attrition API v3", version="1.3.0", lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: EmployeeInput,
    threshold: float = Query(default=DEFAULT_THRESHOLD, gt=0.0, lt=1.0),
    api_key: str = Depends(verify_api_key),
):
    if "pipeline" not in model_state:
        raise HTTPException(status_code=500, detail="Model belum dimuat.")

    raw = payload.model_dump()
    ref = raw.pop("employee_id_ref")
    input_df = pd.DataFrame([{col: raw[col] for col in FEATURE_COLS}])

    prob = float(model_state["pipeline"].predict_proba(input_df)[0][1])
    pred = "Yes" if prob >= threshold else "No"

    record_id = save_prediction(
        attrition_prediction=pred,
        resign_probability=prob,
        input_features=raw,
        employee_id_ref=ref,
        api_key=api_key,
    )
    return PredictionResponse(
        record_id=record_id,
        api_key_masked=mask_api_key(api_key),
        attrition_prediction=pred,
        resign_probability=round(prob, 4),
        saved_to_db=True,
    )


@app.get("/predictions", response_model=list[PredictionLogItem], dependencies=[Depends(verify_api_key)])
def predictions(limit: int = Query(default=50, ge=1, le=500)):
    return get_predictions(limit=limit)


@app.get("/api/dashboard-data", dependencies=[Depends(verify_api_key)])
def dashboard_data(
    job_role: str | None = Query(default=None),
    over_time: str | None = Query(default=None),
    limit: int = Query(default=30, ge=1, le=200),
):
    d_path = Path(os.getenv("DATA_PATH", str(DATA_PATH)))
    df = pd.read_csv(d_path)
    filtered = df.copy()
    if job_role and "JobRole" in filtered.columns:
        filtered = filtered[filtered["JobRole"] == job_role]
    if over_time and "OverTime" in filtered.columns:
        filtered = filtered[filtered["OverTime"] == over_time]

    attrition_by_role = {}
    if {"JobRole", "Attrition"}.issubset(filtered.columns):
        yes_only = filtered[filtered["Attrition"].astype(str).str.strip() == "Yes"]
        attrition_by_role = yes_only["JobRole"].value_counts().to_dict()

    return {
        "job_roles": sorted(df["JobRole"].dropna().unique().tolist()) if "JobRole" in df.columns else [],
        "total_rows": int(len(filtered)),
        "attrition_yes_by_role": attrition_by_role,
        "stats": get_prediction_stats(),
        "logs": get_predictions(limit=limit),
        "rows": filtered.head(100).fillna("-").to_dict(orient="records"),
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(
        """
        <!DOCTYPE html><html lang="id"><head><meta charset="UTF-8"/><title>Attrition Dashboard v3</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
          body{font-family:system-ui,sans-serif;background:#f8fafc;margin:0;color:#0f172a}
          .wrap{max-width:1150px;margin:1.5rem auto;padding:0 1rem}
          .box{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:1rem;margin-bottom:1rem}
          .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:.75rem}
          input,select,button{padding:.45rem .6rem;border:1px solid #cbd5e1;border-radius:6px}
          button{background:#2563eb;color:#fff;border:none;font-weight:600;cursor:pointer}
          table{width:100%;border-collapse:collapse;font-size:.88rem}
          th,td{padding:.55rem .7rem;border-bottom:1px solid #e2e8f0;text-align:left}
          th{background:#f1f5f9}
        </style></head><body><div class="wrap">
          <h2>HR Attrition Dashboard (7 Fitur + SQLite)</h2>
          <div class="box">
            <input id="apiKey" type="password" value="dev-secret-key-123" placeholder="X-API-Key"/>
            <select id="fRole" onchange="loadAll()"><option value="">Semua JobRole</option></select>
            <select id="fOT" onchange="loadAll()"><option value="">Semua OverTime</option><option value="Yes">Yes</option><option value="No">No</option></select>
            <button onclick="loadAll()">Muat Data</button>
          </div>
          <div class="box">
            <strong>Prediksi Karyawan Baru</strong>
            <div class="grid" style="margin-top:.6rem">
              <input id="ref" value="EMP-V3-01" placeholder="Ref"/>
              <input id="age" type="number" value="28" placeholder="Age"/>
              <input id="role" value="Sales Representative" placeholder="JobRole"/>
              <input id="marital" value="Single" placeholder="MaritalStatus"/>
              <input id="income" type="number" value="2800" placeholder="MonthlyIncome"/>
              <input id="years" type="number" value="1" placeholder="YearsAtCompany"/>
              <input id="env" type="number" value="1" min="1" max="4" placeholder="EnvSat (1-4)"/>
              <select id="ot"><option value="Yes">OverTime: Yes</option><option value="No">OverTime: No</option></select>
              <button onclick="predictNow()">Prediksi & Simpan</button>
            </div>
            <div id="msg" style="margin-top:.6rem;color:#1d4ed8;font-weight:600"></div>
          </div>
          <div class="box"><canvas id="chartAttr"></canvas></div>
          <div class="box" style="overflow-x:auto">
            <h4>Riwayat Prediksi (SQLite)</h4>
            <table><thead><tr><th>ID</th><th>Waktu</th><th>Ref</th><th>API Key</th><th>Prediksi</th><th>Prob</th></tr></thead><tbody id="logs"></tbody></table>
          </div>
        </div>
        <script>
          let chart=null, roleInit=false;
          async function predictNow(){
            const key=document.getElementById("apiKey").value;
            const res=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json","X-API-Key":key},body:JSON.stringify({
              employee_id_ref:document.getElementById("ref").value,
              Age:Number(document.getElementById("age").value),
              JobRole:document.getElementById("role").value,
              MaritalStatus:document.getElementById("marital").value,
              MonthlyIncome:Number(document.getElementById("income").value),
              YearsAtCompany:Number(document.getElementById("years").value),
              EnvironmentSatisfaction:Number(document.getElementById("env").value),
              OverTime:document.getElementById("ot").value
            })});
            const d=await res.json();
            document.getElementById("msg").textContent=res.ok
              ? `ID #${d.record_id}: Prediksi=${d.attrition_prediction} (${d.resign_probability}) | API Key=${d.api_key_masked}`
              : `Error: ${d.detail}`;
            loadAll();
          }
          async function loadAll(){
            const key=document.getElementById("apiKey").value;
            const rVal=document.getElementById("fRole").value;
            const oVal=document.getElementById("fOT").value;
            const q=new URLSearchParams();
            if(rVal)q.append("job_role",rVal); if(oVal)q.append("over_time",oVal);
            const res=await fetch("/api/dashboard-data?"+q.toString(),{headers:{"X-API-Key":key}});
            if(!res.ok)return;
            const d=await res.json();
            if(!roleInit){
              const sel=document.getElementById("fRole");
              d.job_roles.forEach(x=>{const o=document.createElement("option");o.value=x;o.textContent=x;sel.appendChild(o);});
              roleInit=true;
            }
            const labels=Object.keys(d.attrition_yes_by_role), vals=Object.values(d.attrition_yes_by_role);
            if(chart)chart.destroy();
            chart=new Chart(document.getElementById("chartAttr"),{type:"bar",data:{labels,datasets:[{label:"Jumlah Resign (Yes) per JobRole",data:vals,backgroundColor:"#ef4444"}]}});
            document.getElementById("logs").innerHTML=d.logs.map(x=>`<tr><td>${x.id}</td><td>${x.created_at.slice(0,19)}</td><td>${x.employee_id_ref||"-"}</td><td><code>${x.api_key_masked}</code></td><td>${x.attrition_prediction}</td><td>${x.resign_probability}</td></tr>`).join("");
          }
          window.addEventListener("DOMContentLoaded",loadAll);
        </script></body></html>
        """
    )
    
@app.get("/api/jobrole-matrix", dependencies=[Depends(verify_api_key)])
def get_jobrole_matrix():
    d_path = Path(os.getenv("DATA_PATH", str(DATA_PATH)))
    df = pd.read_csv(d_path) if d_path.exists() else pd.DataFrame()
    pred_logs = get_predictions(limit=500)

    # Hitung prediksi dari SQLite per JobRole
    pred_by_role: dict[str, dict[str, float]] = {}
    for item in pred_logs:
        role = str(item.get("input_features", {}).get("JobRole") or "Unknown")
        if role not in pred_by_role:
            pred_by_role[role] = {"pred_yes": 0, "pred_no": 0, "prob_sum": 0.0, "pred_total": 0}
        pred_by_role[role]["pred_total"] += 1
        pred_by_role[role]["prob_sum"] += float(item.get("resign_probability") or 0.0)
        if item.get("attrition_prediction") == "Yes":
            pred_by_role[role]["pred_yes"] += 1
        else:
            pred_by_role[role]["pred_no"] += 1

    roles = sorted(df["JobRole"].dropna().unique().tolist()) if "JobRole" in df.columns else []
    for r in pred_by_role:
        if r not in roles:
            roles.append(r)

    matrix_rows = []
    for role in roles:
        sub = df[df["JobRole"] == role] if (not df.empty and "JobRole" in df.columns) else pd.DataFrame()
        actual_yes = (
            int((sub["Attrition"].astype(str).str.strip() == "Yes").sum())
            if (not sub.empty and "Attrition" in sub.columns)
            else 0
        )
        actual_no = (
            int((sub["Attrition"].astype(str).str.strip() == "No").sum())
            if (not sub.empty and "Attrition" in sub.columns)
            else 0
        )
        actual_total = actual_yes + actual_no
        actual_rate_pct = round((actual_yes / actual_total * 100), 2) if actual_total > 0 else 0.0

        p = pred_by_role.get(role, {"pred_yes": 0, "pred_no": 0, "prob_sum": 0.0, "pred_total": 0})
        avg_prob_pct = round((p["prob_sum"] / p["pred_total"] * 100), 2) if p["pred_total"] > 0 else 0.0

        matrix_rows.append(
            {
                "JobRole": role,
                "actual_yes": actual_yes,
                "actual_no": actual_no,
                "actual_total": actual_total,
                "actual_resign_rate_pct": actual_rate_pct,
                "pred_yes": int(p["pred_yes"]),
                "pred_no": int(p["pred_no"]),
                "pred_total": int(p["pred_total"]),
                "avg_pred_prob_pct": avg_prob_pct,
            }
        )

    matrix_rows.sort(key=lambda x: x["actual_resign_rate_pct"], reverse=True)
    return {"matrix": matrix_rows}    
    
    
    