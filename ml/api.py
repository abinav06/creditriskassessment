from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional 

app = FastAPI(title="Credit Risk Assessment API", version="1.0")

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "artifacts" / "credit_risk_pipeline.joblib"

pipeline = None
try:
    pipeline = joblib.load(PIPELINE_PATH)
    print("Pipeline loaded successfully.")
except FileNotFoundError:
    print("FATAL ERROR: Pipeline artifact not found at specified path.")


class LoanApplication(BaseModel):
    loan_amnt: float
    funded_amnt: float
    funded_amnt_inv: float
    term: str
    int_rate: float
    installment: float
    grade: str
    sub_grade: str
    emp_title: Optional[str] = None
    emp_length: Optional[str] = None
    home_ownership: str
    annual_inc: float
    verification_status: str
    pymnt_plan: str
    url: Optional[str] = None
    purpose: str
    title: Optional[str] = None
    zip_code: Optional[str] = None
    addr_state: str
    dti: Optional[float] = None
    delinq_2yrs: float
    fico_range_low: float
    fico_range_high: float
    inq_last_6mths: Optional[float] = None
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: Optional[float] = None
    total_acc: float
    initial_list_status: str
    collections_12_mths_ex_med: Optional[float] = None
    policy_code: float
    application_type: str
    acc_now_delinq: float
    tot_coll_amt: Optional[float] = None
    tot_cur_bal: Optional[float] = None
    total_rev_hi_lim: Optional[float] = None
    acc_open_past_24mths: Optional[float] = None
    avg_cur_bal: Optional[float] = None
    bc_open_to_buy: Optional[float] = None
    bc_util: Optional[float] = None
    chargeoff_within_12_mths: Optional[float] = None
    delinq_amnt: float
    mo_sin_old_il_acct: Optional[float] = None
    mo_sin_old_rev_tl_op: Optional[float] = None
    mo_sin_rcnt_rev_tl_op: Optional[float] = None
    mo_sin_rcnt_tl: Optional[float] = None
    mort_acc: Optional[float] = None
    mths_since_recent_bc: Optional[float] = None
    mths_since_recent_inq: Optional[float] = None
    num_accts_ever_120_pd: Optional[float] = None
    num_actv_bc_tl: Optional[float] = None
    num_actv_rev_tl: Optional[float] = None
    num_bc_sats: Optional[float] = None
    num_bc_tl: Optional[float] = None
    num_il_tl: Optional[float] = None
    num_op_rev_tl: Optional[float] = None
    num_rev_accts: Optional[float] = None
    num_rev_tl_bal_gt_0: Optional[float] = None
    num_sats: Optional[float] = None
    num_tl_120dpd_2m: Optional[float] = None
    num_tl_30dpd: Optional[float] = None
    num_tl_90g_dpd_24m: Optional[float] = None
    num_tl_op_past_12m: Optional[float] = None
    pct_tl_nvr_dlq: Optional[float] = None
    percent_bc_gt_75: Optional[float] = None
    pub_rec_bankruptcies: Optional[float] = None
    tax_liens: Optional[float] = None
    tot_hi_cred_lim: Optional[float] = None
    total_bal_ex_mort: Optional[float] = None
    total_bc_limit: Optional[float] = None
    total_il_high_credit_limit: Optional[float] = None
    hardship_flag: str
    disbursement_method: str
    credit_history_months: float
    loan_to_income_ratio: float
    installment_to_income_ratio: float

@app.post("/predict")
async def predict(application: LoanApplication):
    if pipeline is None:
        return {"error": "Model is not loaded. Cannot make predictions."}

    input_df = pd.DataFrame([application.model_dump()])
    
    try:
        pred_proba = pipeline.predict_proba(input_df)[0, 1]
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}
    
    risk_threshold = 0.5
    is_high_risk = pred_proba > risk_threshold

    return {
        "prediction_probability_default": f"{pred_proba:.4f}",
        "is_high_risk": bool(is_high_risk),
        "risk_threshold": risk_threshold
    }


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Credit Risk API is running"}