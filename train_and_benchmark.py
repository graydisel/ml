import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Инструменты экспорта в формат ONNX
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType as ToolFloatTensorType

# ---------------------------------------------------------------------------
# 1. Загрузка данных и подготовка признаков
# ---------------------------------------------------------------------------
csv_path = "unified_maritime_dataset.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Файл {csv_path} не найден! Сначала выполни generate_multiship_dataset.py.")

df = pd.read_csv(csv_path)

# Входной вектор из 5 универсальных параметров
features = [
    "sog",            # 0: Скорость (узлы)
    "mean_draft",     # 1: Осадка (м)
    "trim",           # 2: Дифферент (м)
    "length",         # 3: Длина корпуса (м)
    "beam",           # 4: Ширина корпуса (м)
    "engine_mcr_kw",  # 5: Паспортный MCR (кВт)
    "hydro_load"      # 6: Физический индекс мощности: ((L * B * draft)^(2/3) * sog^3) / 1000
]
target_col = "fuel_kg_per_h"

df_clean = df.dropna(subset=features + [target_col]).copy()
print(f"✅ Загружено строк датасета: {len(df_clean)}")
print(f"Представленные классы флота: {df_clean['vessel_type'].unique().tolist()}")

X = df_clean[features].astype(np.float32)
y_hourly = df_clean[target_col].astype(np.float32)

# ---------------------------------------------------------------------------
# 2. Разделение: 85% на разработку/CV, 15% — неприкосновенный финальный тест
# ---------------------------------------------------------------------------
X_dev, X_test, y_dev_hourly, y_test_hourly, idx_dev, idx_test = train_test_split(
    X, y_hourly, df_clean.index, test_size=0.15, random_state=42, shuffle=True
)

y_dev_daily = (y_dev_hourly * 24.0) / 1000.0
y_test_daily = (y_test_hourly * 24.0) / 1000.0

print(f"Размер выборки разработки (Train/CV): {len(X_dev)} строк")
print(f"Размер контрольного теста (Holdout Test): {len(X_test)} строк")

# ---------------------------------------------------------------------------
# 3. Конфигурации моделей для мультисудового флота
# ---------------------------------------------------------------------------
MONOTONE_RULES = [1, 1, 0, 0, 0, 0, 1]

models = {
    "LightGBM (Standard)": LGBMRegressor(
        n_estimators=150,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.04,
        min_child_samples=12,
        monotone_constraints=MONOTONE_RULES,
        verbose=-1,
        random_state=42
    ),
    "LightGBM (Regulated)": LGBMRegressor(
        n_estimators=200,
        max_depth=7,
        num_leaves=40,
        learning_rate=0.03,
        reg_alpha=0.3,
        reg_lambda=0.6,
        min_child_samples=10,
        monotone_constraints=MONOTONE_RULES,
        verbose=-1,
        random_state=42
    ),
    "XGBoost (Fleet-Tuned)": XGBRegressor(
        n_estimators=140,
        max_depth=5,
        learning_rate=0.04,
        reg_alpha=0.4,
        reg_lambda=0.8,
        monotone_constraints=tuple(MONOTONE_RULES),
        random_state=42
    )
}

# ---------------------------------------------------------------------------
# 4. Кросс-валидация (5-Fold CV) на выборке разработки
# ---------------------------------------------------------------------------
print("\n" + "="*75)
print(f"{'Модель':<28} | {'CV MAE (т/сут)':<16} | {'CV Ошибка MAPE (%)':<10}")
print("="*75)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_model_name = None
lowest_cv_mape = float("inf")

for name, model in models.items():
    fold_maes = []
    fold_mapes = []
    
    for train_idx, val_idx in kf.split(X_dev):
        X_tr, X_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
        y_tr, y_val = y_dev_hourly.iloc[train_idx], y_dev_hourly.iloc[val_idx]
        y_val_daily = y_dev_daily.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        preds_hourly = model.predict(X_val)
        preds_daily = (preds_hourly * 24.0) / 1000.0
        
        fold_maes.append(mean_absolute_error(y_val_daily, preds_daily))
        fold_mapes.append(mean_absolute_percentage_error(y_val_daily, preds_daily) * 100.0)
        
    avg_mae = np.mean(fold_maes)
    avg_mape = np.mean(fold_mapes)
    print(f"{name:<28} | {avg_mae:<16.2f} | {avg_mape:<10.2f}%")
    
    if avg_mape < lowest_cv_mape:
        lowest_cv_mape = avg_mape
        best_model_name = name

print("="*75)
print(f"🏆 Победитель кросс-валидации: {best_model_name} (CV MAPE: {lowest_cv_mape:.2f}%)")

# ---------------------------------------------------------------------------
# 5. Финальный экзамен победителя на отложенном тесте (Holdout 15%)
# ---------------------------------------------------------------------------
print("\n" + "#"*75)
print("ФИНАЛЬНЫЙ КОНТРОЛЬНЫЙ ТЕСТ НА НЕЗАВИСИМЫХ ДАННЫХ (Holdout Test Set)")
print("#"*75)

final_model = models[best_model_name]
# Обучаем победителя на всех 85% выборки разработки
final_model.fit(X_dev, y_dev_hourly)

# Предсказываем на 15% данных, которые модель никогда не видела
test_preds_hourly = final_model.predict(X_test)
test_preds_daily = (test_preds_hourly * 24.0) / 1000.0

holdout_mae = mean_absolute_error(y_test_daily, test_preds_daily)
holdout_mape = mean_absolute_percentage_error(y_test_daily, test_preds_daily) * 100.0

print(f"Абсолютная средняя ошибка (MAE): {holdout_mae:.2f} т/сутки")
print(f"Относительная ошибка (MAPE):     {holdout_mape:.2f}%")

# Сравнение факта и прогноза на случайных примерах из разных классов
df_results = df_clean.loc[idx_test].copy()
df_results["Actual_Tons_Day"] = y_test_daily.values.round(2)
df_results["Predicted_Tons_Day"] = test_preds_daily.round(2)
df_results["Error_Percent"] = (
    np.abs(df_results["Actual_Tons_Day"] - df_results["Predicted_Tons_Day"]) / df_results["Actual_Tons_Day"] * 100.0
).round(2)

print("\nПримеры предсказаний на тестовых данных:")
sample_table = df_results[[
    "vessel_type", "sog", "mean_draft", "engine_mcr_kw", "Actual_Tons_Day", "Predicted_Tons_Day", "Error_Percent"
]].sample(n=min(8, len(df_results)), random_state=42)
print(sample_table.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Обучение на 100% данных и экспорт в ONNX
# ---------------------------------------------------------------------------
print("\n" + "="*75)
print(f"Экспорт {best_model_name} в ONNX...")

# Переобучаем на всем объеме (100%)
final_model.fit(X, y_hourly)

onnx_output_path = "fuel_model.onnx"
initial_type = [("float_input", FloatTensorType([None, len(features)]))]

try:
    if "LightGBM" in best_model_name:
        onnx_model = onnxmltools.convert_lightgbm(
            final_model, 
            initial_types=[("float_input", ToolFloatTensorType([None, len(features)]))]
        )
    elif "XGBoost" in best_model_name:
        onnx_model = onnxmltools.convert_xgboost(
            final_model, 
            initial_types=[("float_input", ToolFloatTensorType([None, len(features)]))]
        )
    else:
        onnx_model = convert_sklearn(final_model, initial_types=initial_type)

    with open(onnx_output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"🚀 Файл '{onnx_output_path}' успешно сгенерирован и готов к интеграции!")
    print(f"Входной вектор модели ({len(features)} признаков): {features}")
except Exception as e:
    print(f"⚠️ Ошибка конвертации в ONNX: {e}")