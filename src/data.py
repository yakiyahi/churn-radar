import pandas as pd


TARGET = 'Churn'
file='data/Telco-Customer-Churn.csv'
def load_raw(file):
    return pd.read_csv(file)

def clean_data (df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["customerID"])

    # TotalCharges est lu comme du texte (valeurs vides)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce").fillna(0)
    # Harmonisation des modalités redondantes
    df = df.replace({"No internet service": "No", "No phone service": "No"})
    df[TARGET] = (df[TARGET]=='Yes').astype(int)

    return df.drop_duplicates()

if __name__ =="__main__":
    df = clean_data(load_raw(file))
    print(df.head)
   
