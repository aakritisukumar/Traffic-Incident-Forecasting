import pandas as pd
import numpy as np
import os

def load_and_preprocess_data(path="data/dataset.csv"):
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go to project root
    full_path = os.path.join(base_dir, path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")

    df = pd.read_csv(full_path)

    # Drop unnecessary columns
    df.drop([
        'VERAEND_VORMONAT_PROZENT',
        'VERAEND_VORJAHRESMONAT_PROZENT',
        'ZWOELF_MONATE_MITTELWERT'
    ], axis=1, inplace=True)

    for col in ['WERT', 'VORJAHRESWERT']:
        df[col] = df[col].ffill()
        df[col] = df[col].bfill()

    df = df[(df['MONATSZAHL'] == 'Alkoholunfälle') & (df['AUSPRAEGUNG'] == 'insgesamt')]

    annual_data = df[df['MONAT'] == 'Summe'].copy()
    monthly_data = df[df['MONAT'] != 'Summe'].copy()

    monthly_data['MONAT'] = pd.to_datetime(monthly_data['MONAT'], format='%Y%m', errors='coerce')
    monthly_data.dropna(subset=['MONAT'], inplace=True)
    monthly_data.set_index('MONAT', inplace=True)
    monthly_data.sort_index(inplace=True)

    annual_data['JAHR'] = annual_data['JAHR'].astype(int)
    annual_data.set_index('JAHR', inplace=True)
    annual_data.sort_index(inplace=True)

    monthly_data['WERT_log'] = np.log1p(monthly_data['WERT'])

    return df, monthly_data, annual_data


if __name__ == "__main__":
    
    if __name__ == "__main__":
        df, monthly_data, annual_data = load_and_preprocess_data()

    
    print("✅ Preprocessing complete.")
    print("\n📅 Monthly Data Sample:")
    print(monthly_data.head())

    print("\n📈 Annual Data Sample:")
    print(annual_data.head())
