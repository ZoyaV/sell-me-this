"""Автопроверка бейзлайна CTR в look_at_data_task / look_at_data_solution."""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_SHOWS = 10


def _parse_dates(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s)
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def make_group_day(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["Дата"] = _parse_dates(df["Дата"])
    out = df.groupby(["Дата", "Группа"], as_index=False).agg(
        показы=("Показы", "sum"),
        клики=("Клики", "sum"),
    )
    out["CTR"] = np.where(out["показы"] > 0, 100 * out["клики"] / out["показы"], 0.0)
    out["месяц"] = out["Дата"].dt.month
    return out


def split_train_test(group_day: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = group_day.copy()
    work["Дата"] = _parse_dates(work["Дата"])
    work = work.loc[work["показы"] >= MIN_SHOWS]
    train = work.loc[work["Дата"].dt.year == 2023].copy()
    test = work.loc[work["Дата"].dt.year == 2024].copy()
    return train, test


def dummy_ctr(train: pd.DataFrame) -> float:
    return float(100 * train["клики"].sum() / train["показы"].sum())


def check_baseline(group_day: pd.DataFrame, predict_ctr) -> None:
    train, test = split_train_test(group_day)
    assert len(train) > 0 and len(test) > 0, "После фильтра показы ≥ 10 нет 2023 или 2024."

    preds = test.apply(predict_ctr, axis=1).astype(float)
    assert preds.notna().all(), "predict_ctr вернул пустое значение."
    assert ((preds >= 0) & (preds <= 100)).all(), "CTR должен быть от 0 до 100."

    dummy = dummy_ctr(train)
    dummy_mae = float(np.mean(np.abs(dummy - test["CTR"])))
    model_mae = float(np.mean(np.abs(preds - test["CTR"])))
    print(f"константа (CTR поезда) = {dummy:.2f}, MAE = {dummy_mae:.3f}")
    print(f"ваши if'ы, MAE = {model_mae:.3f}")
    assert model_mae < dummy_mae, (
        "Нужно MAE строго меньше, чем у константы «средний CTR 2023». "
        "Среднее каждой из 25 групп на 2023 часто проигрывает: в 2024 детские уже не те. "
        "Попробуйте грубые корзины (детские / города / остальное) и месяц."
    )
    print("Бейзлайн: ок")
