import pandas as pd

def load_and_clean(path: str, nrows: int = None):
    """
    Loads the data, creates the target, engineers new features,
    and performs cleaning, now with leakage removal.
    """
    df = pd.read_csv(path, low_memory=False, nrows=nrows)
    df['target'] = df['loan_status'].apply(lambda x: 1 if x in ['Charged Off', 'Default'] else 0)
    df.drop(columns=['loan_status'], inplace=True)

    leakage_cols = [
        'id', 'member_id', 'recoveries', 'collection_recovery_fee',
        'last_pymnt_amnt', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
        'last_credit_pull_d', 'last_pymnt_d', 'out_prncp', 'out_prncp_inv',
        'total_pymnt', 'total_pymnt_inv',
        'last_fico_range_high',  # LEAKAGE
        'last_fico_range_low',   # LEAKAGE
        'debt_settlement_flag' # LEAKAGE
    ]
    df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True)

    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
    
    df['credit_history_months'] = ((df['issue_d'] - df['earliest_cr_line']).dt.days / 30.44).fillna(0)
    df['loan_to_income_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    df['installment_to_income_ratio'] = df['installment'] / (df['annual_inc'] / 12 + 1)
    df.drop(columns=['issue_d', 'earliest_cr_line'], inplace=True)
    
    df = df.dropna(thresh=len(df) * 0.8, axis=1)
    df = df.select_dtypes(include=['number', 'object'])
    
    return df