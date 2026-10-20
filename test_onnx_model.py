import os
import numpy as np
import onnxruntime as ort

MODEL_PATH = "fuel_model.onnx"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Файл модели '{MODEL_PATH}' не найден! Сначала запусти train_and_benchmark.py.")

# Инициализация ONNX Runtime сессии
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("✅ ONNX сессия успешно запущена.")
print(f"Входной тензор: '{input_name}', Выходной тензор: '{output_name}'")
print(f"Ожидаемая размерность входа: {session.get_inputs()[0].shape}\n")

def calc_hydro_load(length: float, beam: float, draft: float, sog: float) -> float:
    """Расчёт гидродинамической нагрузки по закону подобия."""
    block_vol = length * beam * draft
    return float(((block_vol ** (2.0 / 3.0)) * (sog ** 3.0)) / 1000.0)

def predict_fuel(sog: float, draft: float, trim: float, length: float, beam: float, mcr_kw: float) -> tuple[float, float]:
    """
    Выполняет инференс ONNX модели на 7 признаках:
    [sog, mean_draft, trim, length, beam, engine_mcr_kw, hydro_load]
    """
    h_load = calc_hydro_load(length, beam, draft, sog)
    
    input_vector = np.array([[sog, draft, trim, length, beam, mcr_kw, h_load]], dtype=np.float32)
    result = session.run([output_name], {input_name: input_vector})
    
    # Безопасное извлечение одиночного числа
    kg_per_hour = float(np.array(result[0]).flatten()[0])
    tons_per_day = (kg_per_hour * 24.0) / 1000.0
    return kg_per_hour, tons_per_day

# ---------------------------------------------------------------------------
# ТЕСТ 1: Проверка на реальных судах из базы
# ---------------------------------------------------------------------------
print("=" * 95)
print("ТЕСТ 1: Сверка предсказаний с фактическими данными известных судов")
print("=" * 95)

test_vessels = [
    {
        "name": "MAVERICK (Handysize 38k DWT)",
        "sog": 11.27,
        "draft": 9.80,
        "trim": 1.53,
        "length": 181.0,
        "beam": 30.0,
        "mcr_kw": 6050.0,
        "actual_tons": 12.50
    },
    {
        "name": "Kamsarmax (Zenodo, 82k DWT)",
        "sog": 11.1,
        "draft": 6.33,
        "trim": 3.21,
        "length": 229.0,
        "beam": 32.26,
        "mcr_kw": 9930.0,
        "actual_tons": 13.02
    },
    {
        "name": "Aquavita Trust (Ultramax 63k DWT)",
        "sog": 11.50,
        "draft": 13.30,
        "trim": 0.40,
        "length": 199.98,
        "beam": 32.26,
        "mcr_kw": 7220.0,
        "actual_tons": 19.55
    },
    {
        "name": "Feeder Container (2500 TEU)",
        "sog": 17.00,
        "draft": 10.50,
        "trim": 0.30,
        "length": 208.0,
        "beam": 30.0,
        "mcr_kw": 18500.0,
        "actual_tons": 45.00
    }
]

print(f"{'Судно':<34} | {'Скорость':<8} | {'Осадка':<6} | {'Факт (т/сут)':<12} | {'Модель (т/сут)':<14} | {'Ошибка (%)':<10}")
print("-" * 95)

for v in test_vessels:
    kg_h, tons_day = predict_fuel(v["sog"], v["draft"], v["trim"], v["length"], v["beam"], v["mcr_kw"])
    error_pct = abs(tons_day - v["actual_tons"]) / v["actual_tons"] * 100.0
    print(f"{v['name']:<34} | {v['sog']:<8.2f} | {v['draft']:<6.1f} | {v['actual_tons']:<12.2f} | {tons_day:<14.2f} | {error_pct:<10.1f}%")

# ---------------------------------------------------------------------------
# ТЕСТ 2: Гидродинамическая кривая мощности для Aquavita Trust (MCR: 7220 kW)
# ---------------------------------------------------------------------------
print("\n" + "=" * 95)
print("ТЕСТ 2: Проверка адекватности кривой расхода для Aquavita Trust (L: 199.98m, B: 32.26m, Draft: 13.3m)")
print("=" * 95)
print(f"{'Скорость (узлы)':<16} | {'Расход (кг/час)':<18} | {'Расход (т/сутки)':<18} | {'Прирост на узел':<15}")
print("-" * 95)

prev_tons = None
for speed in np.arange(9.0, 14.5, 1.0):
    kg_h, tons_day = predict_fuel(
        sog=speed, 
        draft=13.30, 
        trim=0.40, 
        length=199.98, 
        beam=32.26, 
        mcr_kw=7220.0
    )
    delta_str = f"+{tons_day - prev_tons:.2f} т" if prev_tons is not None else "-"
    prev_tons = tons_day
    print(f"{speed:<16.1f} | {kg_h:<18.1f} | {tons_day:<18.2f} | {delta_str:<15}")

print("=" * 95)
print("✅ Тестирование успешно завершено!")