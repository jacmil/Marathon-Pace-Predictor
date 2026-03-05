import pandas as pd

def load_vickers():
    """Loads Vickers and Vertosick dataset"""
    df = pd.read_excel("../data/raw/vickers_dataset.xlsx")
    race_times = ["k5_ti_adj", "k10_ti_adj", "m5_ti_adj", "m10_ti_adj", "mh_ti_adj", "mf_ti_adj"]
    df[race_times] = df[race_times] / 60
    return df

def load_useable_vickers():
    """Loads cleaned Vickers and Vertosick database"""
    df = pd.read_excel("../data/raw/vickers_dataset.xlsx")
    race_times = ["k5_ti_adj", "k10_ti_adj", "m5_ti_adj", "m10_ti_adj", "mh_ti_adj", "mf_ti_adj"]
    df[race_times] = df[race_times] / 60
    has_marathon = df[df["mf_ti_adj"].notna()]
    race_cols = ["k5_ti_adj", "k10_ti_adj", "m5_ti_adj", "m10_ti_adj", "mh_ti_adj"]
    has_marathon["other_races"] = has_marathon[race_cols].notna().sum(axis=1)
    has_marathon["other_races"].value_counts().sort_index().to_frame("# of other races")
    useable_data = has_marathon[has_marathon["other_races"] == 2]
    return useable_data