#%%

import lingpy
from lingpy.evaluate.acd import bcubes

import pandas as pd


# %%

datasets = [
            "abvd",
            "afrasian",
            "bai",
            "central_asian",
            "chinese_1964",
            "chinese_2004",
            "huon",
            "ielex",
            "japanese",
            "kadai",
            "kamasau",
            "lolo_burmese",
            "mayan",
            "miao_yao",
            "mixe_zoque",
            "mon_khmer",
            "ob_ugrian",
            "tujia"
            ]
#%%
for db in datasets:
    df = pd.read_table(f"data/inferred/{db}.svmCC.csv", sep=",")
    df["ID"] = range(len(df))
    if 'fullCC' in df.columns:
        df['fullCC'] = df['fullCC'].astype(str)
    else:
        df['fullCC'] = [df.concept.loc[i]+":"+str(df.cc.loc[i]) for i in range(len(df))]
    df = df.rename(columns={"counterpart": "forms", "inferredCC": "lexstatid", "fullCC": "cogid"})
    data_dict = {
        0: df.columns.to_list()
    }
    for i in range(len(df)):
        data_dict[i+1] = df.iloc[i].to_list() 
    wl = lingpy.Wordlist(data_dict, row='concept', col='doculect')
    print(db)
    bcubes(wl, gold="cogid", test="lexstatid")
# %%
