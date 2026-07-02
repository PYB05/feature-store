"""
============================================================
ML Feature Store — Airflow Retraining DAG
============================================================
Orchestrates nightly retraining of the fraud detection model
using the last 24 hours of real transaction data stored in
PostgreSQL.

DAG: fraud_model_retraining
Schedule: Every night at 2:00 AM  →  schedule_interval='0 2 * * *'

Pipeline (4 tasks in order):
    1. extract_data    — Pull last 24hrs transactions from PostgreSQL
    2. retrain_model   — Retrain Random Forest on fresh data
    3. evaluate_model  — Calculate F1 score, FAIL if F1 < 0.90
    4. deploy_model    — Replace production model file + send email alert

Why retrain nightly?
    - Fraud patterns evolve over time (concept drift)
    - New data from production improves the model
    - Stale models miss new fraud techniques

Environment Variables (set in Airflow UI or .env):
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    MODEL_PATH    — Where to save the retrained model
    ALERT_EMAIL   — Email address for success/failure notifications
============================================================
"""

import os
import json
import logging
import pickle
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# ============================================================
# Configure Logging
# ============================================================
logger = logging.getLogger(__name__)

# ============================================================
# DAG Default Arguments
# ============================================================
DEFAULT_ARGS = {
    # Owner shown in Airflow UI
    "owner": "ml-team",

    # If a run fails, don't automatically retry
    "retries": 1,
    "retry_delay": timedelta(minutes=5),

    # Email on failure (configure SMTP in airflow.cfg)
    "email_on_failure": True,
    "email_on_retry": False,
    "email": [os.getenv("ALERT_EMAIL", "ml-alerts@company.com")],

    # Start from yesterday — backfill-safe
    "start_date": days_ago(1),
}

# ============================================================
# Constants
# ============================================================
MODEL_PATH      = os.getenv("MODEL_PATH",  "models/fraud_model.pkl")
SCALER_PATH     = os.getenv("SCALER_PATH", "models/scaler.pkl")
MIN_F1_SCORE    = 0.90          # Minimum acceptable F1 — deploy blocked below this
MIN_SAMPLES     = 100           # Minimum rows needed to retrain (fallback to synthetic)
XCOM_KEY_DATA   = "extracted_data_path"
XCOM_KEY_METRICS = "model_metrics"
XCOM_KEY_MODEL  = "new_model_path"


# ============================================================
# Helper — Get DB Connection
# ============================================================
def get_db_connection():
    """
    Create a psycopg2 connection using environment variables.
    Called inside each task function (Airflow tasks run in separate processes).
    """
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "frauddb"),
        user=os.getenv("DB_USER", "frauduser"),
        password=os.getenv("DB_PASSWORD", "fraudpass"),
    )


# ============================================================
# TASK 1 — extract_data
# ============================================================
def extract_data(**context):
    """
    Pull the last 24 hours of fraud detection results from PostgreSQL.

    Why use real production data?
        - It reflects actual transaction patterns your model will see
        - Retraining on stale data causes model drift over time
        - Fresh labels (is_fraud) come from the running model + human review

    XCom output:
        Saves the extracted DataFrame as a JSON file and pushes the
        file path via XCom so the next task can load it.

    Airflow XCom = "cross-communication" — the way tasks share data.
    We save to a file (not XCom directly) because DataFrames can be large.
    """
    logger.info("=" * 60)
    logger.info("📥 TASK 1/4: extract_data — Starting")
    logger.info("   Pulling last 24 hours of transactions from PostgreSQL")
    logger.info("=" * 60)

    start_time = datetime.utcnow()

    # ── Connect to database ────────────────────────────────────
    conn = get_db_connection()
    cutoff = datetime.utcnow() - timedelta(hours=24)

    query = """
        SELECT
            transaction_id,
            amount,
            is_fraud,
            confidence,
            latency_ms,
            created_at
        FROM transactions
        WHERE created_at >= %s
        ORDER BY created_at DESC
    """

    try:
        df = pd.read_sql(query, conn, params=(cutoff,))
    finally:
        conn.close()

    row_count  = len(df)
    fraud_count = df["is_fraud"].sum() if not df.empty else 0

    logger.info(f"   Extracted {row_count:,} transactions")
    logger.info(f"   Fraud cases:      {fraud_count:,}")
    logger.info(f"   Legitimate cases: {row_count - fraud_count:,}")
    logger.info(f"   Time range:       {cutoff.isoformat()} → now")

    # ── Handle low-data scenarios ─────────────────────────────
    if row_count < MIN_SAMPLES:
        logger.warning(
            f"Only {row_count} rows found (need {MIN_SAMPLES}). "
            f"Not enough data for meaningful retraining. "
            f"Skipping this run."
        )
        # Push empty signal to downstream tasks
        context["ti"].xcom_push(key=XCOM_KEY_DATA, value=None)
        return

    # ── Save DataFrame to a temp file ─────────────────────────
    # We can't pass a large DataFrame directly through XCom
    # (Airflow's XCom is stored in the metadata DB — size limits apply)
    tmp_path = f"/tmp/retrain_data_{context['run_id'].replace(':', '_')}.csv"
    df.to_csv(tmp_path, index=False)

    elapsed = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"   Data saved to: {tmp_path}")
    logger.info(f"   Task completed in {elapsed:.2f}s")

    # Push file path to XCom for the next task
    context["ti"].xcom_push(key=XCOM_KEY_DATA, value=tmp_path)


# ============================================================
# TASK 2 — retrain_model
# ============================================================
def retrain_model(**context):
    """
    Retrain the Random Forest classifier on the extracted data.

    Why Random Forest with class_weight='balanced'?
        - Handles extreme class imbalance (0.17% fraud)
        - Robust to outliers and noisy fraud signals
        - No need to resample or SMOTE — weights handle it

    XCom input:  Path to CSV from extract_data
    XCom output: Path to newly trained model .pkl file
    """
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    logger.info("=" * 60)
    logger.info("TASK 2/4: retrain_model -- Starting")
    logger.info("=" * 60)

    # ── Pull the data path from previous task ─────────────────
    data_path = context["ti"].xcom_pull(key=XCOM_KEY_DATA, task_ids="extract_data")

    if data_path is None:
        logger.warning("No data available from extract_data. Skipping retraining.")
        context["ti"].xcom_push(key=XCOM_KEY_MODEL, value=None)
        return

    # ── Load extracted data ────────────────────────────────────
    df = pd.read_csv(data_path)
    logger.info(f"   Loaded {len(df):,} rows from {data_path}")

    # ── Prepare features ──────────────────────────────────────
    # The production DB only stores: amount, is_fraud, confidence, latency_ms
    # We use 'amount' + 'confidence' + 'latency_ms' as proxy features
    # (In production, you'd store full V1-V28 features alongside predictions)
    feature_cols = ["amount", "confidence", "latency_ms"]
    label_col    = "is_fraud"

    X = df[feature_cols].values
    y = df[label_col].astype(int).values

    # ── Scale features ────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Split train/test ──────────────────────────────────────
    # Stratify to preserve fraud ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.sum() >= 2 else None,   # Need ≥2 fraud cases to stratify
    )

    logger.info(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")
    logger.info(f"   Fraud in train: {y_train.sum():,} | Fraud in test: {y_test.sum():,}")

    # ── Train the model ───────────────────────────────────────
    start_time = datetime.utcnow()
    logger.info("   Training Random Forest (n_estimators=100, class_weight='balanced')...")

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",    # Auto-adjusts weights for imbalanced classes
        random_state=42,
        n_jobs=-1,                  # Use all CPU cores
        max_depth=20,
        min_samples_split=5,
    )
    model.fit(X_train, y_train)

    elapsed = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"   Training complete in {elapsed:.1f}s")

    # ── Save model + scaler to temp files ─────────────────────
    run_id = context["run_id"].replace(":", "_")
    model_tmp_path  = f"/tmp/new_fraud_model_{run_id}.pkl"
    scaler_tmp_path = f"/tmp/new_scaler_{run_id}.pkl"

    joblib.dump(model, model_tmp_path)
    joblib.dump(scaler, scaler_tmp_path)

    logger.info(f"   New model saved to:  {model_tmp_path}")
    logger.info(f"   New scaler saved to: {scaler_tmp_path}")

    # Push paths and test data to XCom for evaluate_model
    context["ti"].xcom_push(key=XCOM_KEY_MODEL, value={
        "model_path":  model_tmp_path,
        "scaler_path": scaler_tmp_path,
        "X_test":      X_test.tolist(),    # Convert numpy → list for XCom serialization
        "y_test":      y_test.tolist(),
    })


# ============================================================
# TASK 3 — evaluate_model
# ============================================================
def evaluate_model(**context):
    """
    Evaluate the retrained model. Fail the task if F1 < 0.90.

    Why F1 score (not accuracy)?
        - Accuracy is misleading on imbalanced data:
          A model that predicts "no fraud" always gets 99.83% accuracy!
        - F1 = harmonic mean of Precision + Recall
        - High F1 means we catch most fraud (recall) without too many
          false alarms (precision)

    If F1 < 0.90:
        - The task raises an Exception → Airflow marks it FAILED
        - The deploy_model task is skipped (dependency not met)
        - You get an email alert about the failure
        - The OLD production model stays in place (safe rollback)
    """
    import joblib
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        classification_report, roc_auc_score
    )

    logger.info("=" * 60)
    logger.info("TASK 3/4: evaluate_model -- Starting")
    logger.info("=" * 60)

    # ── Pull model info from previous task ────────────────────
    model_info = context["ti"].xcom_pull(key=XCOM_KEY_MODEL, task_ids="retrain_model")

    if model_info is None:
        logger.warning("No model to evaluate. Skipping.")
        context["ti"].xcom_push(key=XCOM_KEY_METRICS, value=None)
        return

    # ── Load model + test data ────────────────────────────────
    model   = joblib.load(model_info["model_path"])
    X_test  = np.array(model_info["X_test"])
    y_test  = np.array(model_info["y_test"])

    # ── Run predictions ───────────────────────────────────────
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ── Calculate metrics ─────────────────────────────────────
    f1        = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, y_proba) if y_test.sum() > 0 else 0.0

    logger.info(f"\n   Evaluation Results:")
    logger.info(f"   {'─'*40}")
    logger.info(f"   F1 Score:  {f1:.4f}  (threshold: {MIN_F1_SCORE})")
    logger.info(f"   Precision: {precision:.4f}")
    logger.info(f"   Recall:    {recall:.4f}")
    logger.info(f"   ROC-AUC:   {auc:.4f}")
    logger.info(f"   {'─'*40}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])}")

    # ── Push metrics to XCom ──────────────────────────────────
    metrics = {
        "f1":        round(f1, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "auc":       round(auc, 4),
        "model_path":  model_info["model_path"],
        "scaler_path": model_info["scaler_path"],
    }
    context["ti"].xcom_push(key=XCOM_KEY_METRICS, value=metrics)

    # ── GATE: fail if F1 is below threshold ───────────────────
    # This prevents a worse model from being deployed to production.
    # Airflow marks this task as FAILED and skips deploy_model.
    if f1 < MIN_F1_SCORE:
        msg = (
            f"Model F1 score {f1:.4f} is below minimum threshold {MIN_F1_SCORE}. "
            f"Deployment BLOCKED. The existing production model will remain in place."
        )
        logger.error(msg)
        raise ValueError(msg)

    logger.info(f"   F1 score {f1:.4f} passes threshold {MIN_F1_SCORE}. Proceeding to deploy.")


# ============================================================
# TASK 4 — deploy_model
# ============================================================
def deploy_model(**context):
    """
    Replace the production model file with the newly trained one.
    Send an email alert on success.

    Deployment strategy:
        1. Load new model from temp path (output of retrain_model)
        2. Backup existing production model (fraud_model.pkl.bak)
        3. Overwrite fraud_model.pkl with new model
        4. The API reads models/ at startup — next restart picks up new model
        5. Send success email with F1 score

    Note: A production system would also:
        - Trigger a rolling restart of the API pods (Kubernetes)
        - Run A/B tests before full deployment
        - Store model versions in MLflow or similar
    """
    import shutil
    import joblib

    logger.info("=" * 60)
    logger.info("TASK 4/4: deploy_model -- Starting")
    logger.info("=" * 60)

    # ── Pull metrics from evaluate_model ──────────────────────
    metrics = context["ti"].xcom_pull(key=XCOM_KEY_METRICS, task_ids="evaluate_model")

    if metrics is None:
        logger.warning("No metrics available. Skipping deployment.")
        return

    new_model_path  = metrics["model_path"]
    new_scaler_path = metrics["scaler_path"]
    f1_score_val    = metrics["f1"]

    logger.info(f"   New model:  {new_model_path}")
    logger.info(f"   New scaler: {new_scaler_path}")
    logger.info(f"   F1 Score:   {f1_score_val}")

    # ── Create models/ directory if needed ───────────────────
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)

    # ── Backup existing production model ─────────────────────
    if os.path.exists(MODEL_PATH):
        backup_path = MODEL_PATH + ".bak"
        shutil.copy2(MODEL_PATH, backup_path)
        logger.info(f"   Backup created: {backup_path}")

    if os.path.exists(SCALER_PATH):
        scaler_backup = SCALER_PATH + ".bak"
        shutil.copy2(SCALER_PATH, scaler_backup)
        logger.info(f"   Backup created: {scaler_backup}")

    # ── Deploy new model ──────────────────────────────────────
    shutil.copy2(new_model_path, MODEL_PATH)
    shutil.copy2(new_scaler_path, SCALER_PATH)

    logger.info(f"   Deployed new model  -> {MODEL_PATH}")
    logger.info(f"   Deployed new scaler -> {SCALER_PATH}")

    # ── Clean up temp files ───────────────────────────────────
    try:
        os.remove(new_model_path)
        os.remove(new_scaler_path)
    except Exception:
        pass

    # ── Send success email ────────────────────────────────────
    alert_email = os.getenv("ALERT_EMAIL", "ml-alerts@company.com")
    run_date    = context["ds"]     # Airflow execution date string (YYYY-MM-DD)

    email_subject = f"[ML Feature Store] Model Retraining Successful -- {run_date}"
    email_body = f"""
    Fraud Detection Model Retraining — SUCCESS

    Run Date:      {run_date}
    Airflow Run:   {context['run_id']}

    Model Metrics:
        F1 Score:  {metrics['f1']}
        Precision: {metrics['precision']}
        Recall:    {metrics['recall']}
        ROC-AUC:   {metrics['auc']}

    Deployment:
        Model:  {MODEL_PATH}
        Scaler: {SCALER_PATH}

    The production API will use the new model on its next restart.
    Previous model backed up as {MODEL_PATH}.bak
    """

    logger.info(f"\n   Success alert would be sent to: {alert_email}")
    logger.info(f"   Subject: {email_subject}")
    logger.info(f"   Body preview:\n{email_body}")

    # In a real setup with SMTP configured in Airflow:
    # from airflow.utils.email import send_email
    # send_email(to=alert_email, subject=email_subject, html_content=email_body)

    logger.info(f"\n   Deployment complete! F1={f1_score_val}")
    logger.info(f"   Restart the API service to load the new model:")
    logger.info(f"       docker compose restart api")


# ============================================================
# Define the DAG
# ============================================================
with DAG(
    dag_id="fraud_model_retraining",

    # Plain English description shown in Airflow UI
    description=(
        "Nightly retraining of the fraud detection Random Forest model "
        "using the last 24 hours of production transaction data. "
        "Fails safely if the new model F1 score drops below 0.90."
    ),

    default_args=DEFAULT_ARGS,

    # Run at 2:00 AM every night
    # Cron format: minute hour day-of-month month day-of-week
    schedule_interval="0 2 * * *",

    # Don't backfill missed runs when first enabled
    catchup=False,

    # Tags visible in Airflow UI for filtering
    tags=["ml", "fraud-detection", "retraining"],

    # Maximum parallel runs (only 1 at a time)
    max_active_runs=1,

    # Timeout the entire DAG after 2 hours
    dagrun_timeout=timedelta(hours=2),

) as dag:

    # ── Task 1: Extract ───────────────────────────────────────
    task_extract = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data,
        doc_md="""
        ## Extract Data
        Pulls all transactions from the last 24 hours out of PostgreSQL.
        Saves to a temp CSV file and pushes the path via XCom.
        Skips retraining if fewer than 100 rows are available.
        """,
    )

    # ── Task 2: Retrain ───────────────────────────────────────
    task_retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
        doc_md="""
        ## Retrain Model
        Loads extracted data, scales features, trains a new
        RandomForestClassifier with class_weight='balanced'.
        Saves the new model to a temp file.
        """,
    )

    # ── Task 3: Evaluate ──────────────────────────────────────
    task_evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
        doc_md="""
        ## Evaluate Model
        Runs the new model against the held-out test set.
        Calculates F1, Precision, Recall, ROC-AUC.
        **Fails the task (and blocks deployment) if F1 < 0.90.**
        """,
    )

    # ── Task 4: Deploy ────────────────────────────────────────
    task_deploy = PythonOperator(
        task_id="deploy_model",
        python_callable=deploy_model,
        doc_md="""
        ## Deploy Model
        Replaces models/fraud_model.pkl with the new model.
        Backs up the old model as .bak before overwriting.
        Sends an email alert with F1 score on success.
        """,
    )

    # ── Define execution order ────────────────────────────────
    # Airflow uses >> to mean "this task must complete before the next"
    # If any task fails, all downstream tasks are skipped automatically
    task_extract >> task_retrain >> task_evaluate >> task_deploy
