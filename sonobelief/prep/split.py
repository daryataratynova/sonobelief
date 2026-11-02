"""Train / val / test split of dataset_valid.csv with matched body representation.

Stratified on adiposity_tertile x sex, so each split gets the same mix of lean / mid / high
adiposity men and women. Among N_TRIES stratified random splits we keep the one whose
continuous body measures (fat_frac, waist, age, liver and gallbladder volume) differ least
between splits (smallest max standardized mean difference vs. the whole dataset).

Writes column `ab_split` into work/dataset_valid.csv (the original TotalSegmentator `split`
column is left untouched), work/splits/{train,val,test}.txt, and prints a balance table. Re-running gives the same split.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sonobelief import paths

DATA_OUT = paths.WORK
INTER = os.path.join(DATA_OUT, "intermediate")
FRAC = {"train": 0.80, "val": 0.10, "test": 0.10}
N_TRIES = 500
SEED = 0
BALANCE = ["fat_frac", "waist_circ_cm", "fat_area_cm2", "age", "liver_ml", "gallbladder_ml", "l3_truncated", "noisy"]


def split_once(df, strata, rs):
    tr, rest = train_test_split(df.index.to_numpy(object), train_size=FRAC["train"], stratify=strata, random_state=rs)
    va, te = train_test_split(rest, test_size=FRAC["test"] / (FRAC["val"] + FRAC["test"]),
                              stratify=strata[rest], random_state=rs)
    s = pd.Series("train", index=df.index)
    s[va], s[te] = "val", "test"
    return s


def imbalance(df, s):
    x = df[BALANCE].astype(float)
    mu, sd = x.mean(), x.std()
    return max((abs(x[s == k].mean() - mu) / sd).max() for k in FRAC)


def main():
    path = os.path.join(DATA_OUT, "dataset_valid.csv")
    df = pd.read_csv(path, index_col="study")
    sex = df.gender.fillna("m")  # 1 study with unknown sex, folded into the larger group
    strata = df.adiposity_tertile.astype(str) + "_" + sex

    rng = np.random.default_rng(SEED)
    best = min((split_once(df, strata, int(rs)) for rs in rng.integers(0, 2**31, N_TRIES)),
               key=lambda s: imbalance(df, s))
    df["ab_split"] = best
    df.to_csv(path)
    os.makedirs(os.path.join(DATA_OUT, "splits"), exist_ok=True)
    for k in FRAC:  # one study id per line, for training code that doesn't want the whole table
        with open(os.path.join(DATA_OUT, "splits", f"{k}.txt"), "w") as f:
            f.write("\n".join(sorted(df.index[df.ab_split == k])) + "\n")

    order = list(FRAC)
    print(f"max standardized mean difference: {imbalance(df, best):.3f}\n")
    print(df.ab_split.value_counts().reindex(order).to_string(), "\n")
    print(pd.crosstab(df.adiposity_tertile, df.ab_split, normalize="columns").reindex(columns=order).round(2).to_string(), "\n")
    print(pd.crosstab(sex, df.ab_split, normalize="columns").reindex(columns=order).round(2).to_string(), "\n")
    print(df[BALANCE].astype(float).groupby(df.ab_split).mean().reindex(order).round(2).T.to_string())


if __name__ == "__main__":
    main()
