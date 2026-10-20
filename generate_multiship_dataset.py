import numpy as np
import pandas as pd
import os

# ---------------------------------------------------------------------------
# 1. Синтетические профили флота
# ---------------------------------------------------------------------------
FLEET_PROFILES = {
    # MAVERICK (Handysize 38k DWT — точный паспорт)
    "Handysize_Bulker": {
        "length": 181.0,
        "beam": 30.0,
        "draft_min": 5.5,
        "draft_max": 10.621,
        "disp_min": 15000.0,
        "disp_max": 46718.8,
        "mcr_kw": 6050.0,
        "c_adm": 500.0,
        "speed_min": 9.5,
        "speed_max": 13.5,
        "samples": 400
    },

    # Фидерный контейнеровоз (~2500 TEU)
    "Feeder_Container_2500TEU": {
        "length": 208.0,
        "beam": 30.0,
        "draft_min": 7.2,
        "draft_max": 11.5,
        "disp_min": 21000.0,
        "disp_max": 45000.0,
        "mcr_kw": 18500.0,
        "c_adm": 460.0,
        "speed_min": 12.0,
        "speed_max": 19.5,
        "samples": 400
    },

    # Большой океанский контейнеровоз (~10000 TEU)
    "Large_Container_10000TEU": {
        "length": 336.0,
        "beam": 48.2,
        "draft_min": 9.5,
        "draft_max": 15.5,
        "disp_min": 52000.0,
        "disp_max": 135000.0,
        "mcr_kw": 54000.0,
        "c_adm": 440.0,
        "speed_min": 13.0,
        "speed_max": 22.0,
        "samples": 400
    },

    # Ro-Ro паром
    "RoRo_Vessel": {
        "length": 196.0,
        "beam": 26.5,
        "draft_min": 5.6,
        "draft_max": 7.4,
        "disp_min": 15000.0,
        "disp_max": 27500.0,
        "mcr_kw": 21600.0,
        "c_adm": 420.0,
        "speed_min": 13.5,
        "speed_max": 21.0,
        "samples": 400
    }
}

def calculate_sfoc(load_fraction: float) -> float:
    """Удельный расход топлива (SFOC) в зависимости от % нагрузки MCR."""
    if load_fraction < 0.35:
        return float(np.interp(load_fraction, [0.15, 0.35], [225.0, 185.0]))
    elif load_fraction <= 0.85:
        return float(np.interp(load_fraction, [0.35, 0.70, 0.85], [185.0, 167.0, 171.0]))
    else:
        return float(np.interp(load_fraction, [0.85, 1.0], [171.0, 180.0]))

def calc_hydro_load(length, beam, draft, sog):
    """Векторизованный расчёт гидродинамической нагрузки."""
    block_vol = length * beam * draft
    return ((block_vol ** (2.0 / 3.0)) * (sog ** 3.0)) / 1000.0

def generate_synthetic_fleet() -> pd.DataFrame:
    np.random.seed(42)
    records = []

    for v_name, p in FLEET_PROFILES.items():
        for _ in range(p["samples"]):
            sog = float(np.random.uniform(p["speed_min"], p["speed_max"]))
            draft = float(np.random.uniform(p["draft_min"], p["draft_max"]))
            trim = float(np.random.uniform(-0.3, 1.2))
            
            load_factor = (draft - p["draft_min"]) / (p["draft_max"] - p["draft_min"])
            disp = p["disp_min"] + load_factor * (p["disp_max"] - p["disp_min"])
            
            power_calm = (disp ** (2.0 / 3.0) * (sog ** 3.0)) / p["c_adm"]
            trim_penalty = 1.0 + (abs(trim - 0.4) * 0.02)
            weather_noise = float(np.random.normal(1.0, 0.04))
            
            actual_power = min(power_calm * trim_penalty * weather_noise, p["mcr_kw"] * 0.95)
            load_fraction = actual_power / p["mcr_kw"]
            sfoc = calculate_sfoc(load_fraction)
            
            fuel_kg_h = (actual_power * sfoc) / 1000.0
            me_tons_day = (fuel_kg_h * 24.0) / 1000.0
            h_load = calc_hydro_load(p["length"], p["beam"], draft, sog)

            records.append({
                "vessel_type": v_name,
                "length": p["length"],
                "beam": p["beam"],
                "engine_mcr_kw": p["mcr_kw"],
                "sog": round(sog, 2),
                "mean_draft": round(draft, 2),
                "trim": round(trim, 2),
                "hydro_load": round(float(h_load), 2),
                "fuel_kg_per_h": round(fuel_kg_h, 2),
                "me_consumption_tons": round(me_tons_day, 2),
                "is_synthetic": 1
            })

    return pd.DataFrame(records)

def load_zenodo_kamsarmax(filepath: str, max_samples: int = 500) -> pd.DataFrame:
    if not os.path.exists(filepath):
        print(f"⚠️ Файл Zenodo '{filepath}' не найден.")
        return pd.DataFrame()

    print(f"Загрузка Zenodo из '{filepath}'...")
    df_zenodo = pd.read_csv(filepath) if filepath.endswith(".csv") else pd.read_excel(filepath)

    mask = (
        (df_zenodo['Speed over ground (kn)'] >= 8.5) &
        (df_zenodo['Main engine shaft power (KW)'] >= 1000) &
        (df_zenodo['Main engine consumption (L/hr)'] >= 250)
    )
    df_clean = df_zenodo[mask].copy()

    fuel_kg_h = df_clean['Main engine consumption (L/hr)'] * 0.89
    me_tons_day = (fuel_kg_h * 24.0) / 1000.0
    trim = df_clean['Draft Aft (m)'] - df_clean['Draft Fwd (m)']
    
    length = 229.0
    beam = 32.26
    h_load = calc_hydro_load(length, beam, df_clean['Mean draft (m)'], df_clean['Speed over ground (kn)'])

    result = pd.DataFrame({
        "vessel_type": "Kamsarmax_Zenodo_Real",
        "length": length,
        "beam": beam,
        "engine_mcr_kw": 9930.0,
        "sog": df_clean['Speed over ground (kn)'].round(2),
        "mean_draft": df_clean['Mean draft (m)'].round(2),
        "trim": trim.round(2),
        "hydro_load": h_load.round(2),
        "fuel_kg_per_h": fuel_kg_h.round(2),
        "me_consumption_tons": me_tons_day.round(2),
        "is_synthetic": 0
    })

    if len(result) > max_samples:
        result = result.sample(n=max_samples, random_state=42)

    return result

def main():
    datasets = []

    # А. Aquavita Trust (Noon Reports)
    real_csv = "processed_noon_dataset.csv"
    if os.path.exists(real_csv):
        df_real = pd.read_csv(real_csv)
        print(f"Загрузка Aquavita Trust: {len(df_real)} строк.")
        
        draft_clean = df_real["mean_draft"].bfill().ffill()
        trim_clean = df_real["trim"].bfill().ffill()
        l_aqua, b_aqua = 199.98, 32.26
        
        # Векторизованный расчёт hydro_load без ошибок
        h_load_real = calc_hydro_load(l_aqua, b_aqua, draft_clean, df_real["sog"])

        df_real_subset = pd.DataFrame({
            "vessel_type": "Ultramax_Aquavita_Trust",
            "length": l_aqua,
            "beam": b_aqua,
            "engine_mcr_kw": 7220.0,
            "sog": df_real["sog"].round(2),
            "mean_draft": draft_clean.round(2),
            "trim": trim_clean.round(2),
            "hydro_load": h_load_real.round(2),
            "fuel_kg_per_h": df_real["fuel_kg_per_h"].round(2),
            "me_consumption_tons": df_real["me_consumption_tons"].round(2),
            "is_synthetic": 0
        })
        datasets.append(df_real_subset)

        # Калибровочные синтетические точки (Slow Steaming и Full Speed)
        aquavita_extra = []
        for _ in range(400):
            sog_ext = float(np.random.choice([
                np.random.uniform(8.5, 10.5),
                np.random.uniform(12.5, 14.5)
            ]))
            draft_ext = float(np.random.uniform(6.0, 13.36))
            trim_ext = float(np.random.uniform(-0.2, 0.8))
            
            disp_ext = 20500.0 + ((draft_ext - 6.0) / (13.36 - 6.0)) * (73084.0 - 20500.0)
            p_kw = min((disp_ext ** (2.0 / 3.0) * (sog_ext ** 3.0)) / 500.0, 7220.0 * 0.95)
            load_fraction = p_kw / 7220.0
            sfoc = calculate_sfoc(load_fraction)
            fuel_h = (p_kw * sfoc) / 1000.0
            h_load_ext = calc_hydro_load(l_aqua, b_aqua, draft_ext, sog_ext)

            aquavita_extra.append({
                "vessel_type": "Ultramax_Aquavita_Trust",
                "length": l_aqua,
                "beam": b_aqua,
                "engine_mcr_kw": 7220.0,
                "sog": round(sog_ext, 2),
                "mean_draft": round(draft_ext, 2),
                "trim": round(trim_ext, 2),
                "hydro_load": round(float(h_load_ext), 2),
                "fuel_kg_per_h": round(fuel_h, 2),
                "me_consumption_tons": round((fuel_h * 24.0) / 1000.0, 2),
                "is_synthetic": 1
            })
        datasets.append(pd.DataFrame(aquavita_extra))

    # Б. Zenodo Kamsarmax
    zenodo_file = "kamsarmax_reports.csv" if os.path.exists("kamsarmax_reports.csv") else "New Microsoft Excel Worksheet.xlsx"
    df_zenodo = load_zenodo_kamsarmax(zenodo_file, max_samples=500)
    if not df_zenodo.empty:
        datasets.append(df_zenodo)

    # В. Остальной флот
    print("Генерация профилей флота...")
    df_synth = generate_synthetic_fleet()
    datasets.append(df_synth)

    # Объединение и сохранение
    df_master = pd.concat(datasets, ignore_index=True)
    output_path = "unified_maritime_dataset.csv"
    df_master.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print(f"✅ Датасет готов: '{output_path}'. Всего строк: {len(df_master)}")
    print("="*80)

if __name__ == "__main__":
    main()