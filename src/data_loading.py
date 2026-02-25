import pandas as pd

def load_vickers():
    df = pd.read_excel("../data/raw/vickers_dataset.xlsx")
    race_times = ["k5_ti_adj", "k10_ti_adj", "m5_ti_adj", "m10_ti_adj", "mh_ti_adj", "mf_ti_adj"]
    df[race_times] = df[race_times] / 60
    return df

