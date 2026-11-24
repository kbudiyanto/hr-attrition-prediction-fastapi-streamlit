import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

DB_PATH = "attrition_v3.db"


def mask_api_key(api_key: str | None) -> str:
    if not api_key:
        return "-"
    cleaned = api_key.strip()
    if len(cleaned) <= 4:
        return "*" * len(cleaned)
    return f"{cleaned[:2]}{'*' * (len(cleaned) - 4)}{cleaned[-2:]}"


def init_db(db_path: str = DB_PATH) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                employee_id_ref TEXT,
                api_key_masked TEXT NOT NULL DEFAULT '-',
                attrition_prediction TEXT NOT NULL,
                resign_probability REAL NOT NULL,
                input_features_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_prediction(
    attrition_prediction: str,
    resign_probability: float,
    input_features: dict[str, Any],
    employee_id_ref: str | None = None,
    api_key: str | None = None,
    db_path: str = DB_PATH,
) -> int:
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO prediction_logs (
                created_at,
                employee_id_ref,
                api_key_masked,
                attrition_prediction,
                resign_probability,
                input_features_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                employee_id_ref,
                mask_api_key(api_key),
                attrition_prediction,
                round(float(resign_probability), 6),
                json.dumps(input_features, ensure_ascii=False),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_predictions(limit: int = 50, db_path: str = DB_PATH) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, employee_id_ref, api_key_masked,
                   attrition_prediction, resign_probability, input_features_json
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "employee_id_ref": row["employee_id_ref"],
            "api_key_masked": row["api_key_masked"],
            "attrition_prediction": row["attrition_prediction"],
            "resign_probability": row["resign_probability"],
            "input_features": json.loads(row["input_features_json"]),
        }
        for row in rows
    ]


def get_prediction_stats(db_path: str = DB_PATH) -> dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN attrition_prediction = 'Yes' THEN 1 ELSE 0 END) AS yes_cnt,
                SUM(CASE WHEN attrition_prediction = 'No' THEN 1 ELSE 0 END) AS no_cnt,
                AVG(resign_probability) AS avg_prob
            FROM prediction_logs
            """
        ).fetchone()

    return {
        "total_predictions": int(row[0] or 0),
        "resign_yes_count": int(row[1] or 0),
        "resign_no_count": int(row[2] or 0),
        "avg_resign_prob": round(float(row[3] or 0.0), 4),
    }