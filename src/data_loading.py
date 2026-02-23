import pandas as pd

def load_vickers():
    df = pd.read_excel("../data/raw/vickers_dataset.xlsx")
    return df

