"""Split bycicles.csv into train.csv / test.csv / test_answers.csv.

Test is balanced on the target: half the rows have CTR == 0, half have CTR != 0.
Only CTR is kept as a metric; clicks, shows, spend and conversions are dropped.
test.csv has features only. test_answers.csv is Id, Usage (all Public), CTR.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "bycicles.csv"

DROP = [
    "Ссылка",
    "Показы",
    "Взвешенные показы",
    "Клики",
    "wCTR (%)",
    "Расход (руб.)",
    "Конверсия (%)/Товар добавлен в корзину",
    "Конверсия (%)/Заказ оформлен",
    "Конверсия (%)/Лид с Centra Market//Динамика",
    "Конверсии/Товар добавлен в корзину",
    "Конверсии/Заказ оформлен",
    "Конверсии/Лид с Centra Market//Динамика",
]


def main() -> None:
    raw = pd.read_csv(SOURCE)
    raw.index.name = "row_id"
    out = raw.drop(columns=[c for c in DROP if c in raw.columns]).rename(columns={"CTR (%)": "CTR"})

    zero = out.index[out["CTR"] == 0]
    nonzero = out.index[out["CTR"] != 0]
    n = int(min(len(zero), len(nonzero)) // 2)
    rng = np.random.default_rng(SEED)
    test_zero = rng.choice(zero.to_numpy(), size=n, replace=False)
    test_nz = rng.choice(nonzero.to_numpy(), size=n, replace=False)
    test_idx = np.sort(np.concatenate([test_zero, test_nz]))
    train_idx = np.sort(np.setdiff1d(out.index.to_numpy(), test_idx, assume_unique=False))

    train = out.loc[train_idx].copy()
    test = out.loc[test_idx].copy()
    train.insert(0, "Id", train.index.astype(int))
    test.insert(0, "Id", test.index.astype(int))
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    feature_cols = [c for c in train.columns if c not in ("Id", "CTR")]
    train = train[["Id", "CTR"] + feature_cols]
    train.to_csv(HERE / "train.csv", index=False)
    test[["Id"] + feature_cols].to_csv(HERE / "test.csv", index=False)

    answers = test[["Id"]].copy()
    answers["Usage"] = "Public"
    answers["CTR"] = test["CTR"].to_numpy()
    answers.to_csv(HERE / "test_answers.csv", index=False)

    split = {
        "source": "dataset/bycicles.csv",
        "seed": SEED,
        "target": "CTR",
        "n_per_class_test": n,
        "train_indices": [int(i) for i in train_idx],
        "test_indices": [int(i) for i in test_idx],
        "public_ids": [int(i) for i in answers["Id"]],
    }
    (HERE / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"train {len(train)}  test {len(test)}  ({n} zero + {n} nonzero CTR)")
    print("test CTR==0", float((test["CTR"] == 0).mean()))
    print(answers["Usage"].value_counts().to_dict())
    print("wrote train.csv, test.csv (no target), test_answers.csv")


if __name__ == "__main__":
    main()
