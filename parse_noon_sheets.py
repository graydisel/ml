"""
Парсер суточных судовых отчетов (Noon Reports) из Excel книги daily_reports.xlsx
с адаптивным поиском ячеек и эвристиками под меняющуюся структуру 2024 и 2025 годов.
"""

import re
import sys
from pathlib import Path
import openpyxl
import pandas as pd

# Обеспечиваем корректный вывод UTF-8 в терминале Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def to_float(val, default=None):
    """Преобразует значение ячейки в float, поддерживая запятые и регулярные выражения."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip().replace(',', '.')
    match = re.search(r'[-+]?\d+(?:\.\d+)?', val_str)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return default
    return default


def parse_coordinate(coord_str: str):
    """
    Адаптивный парсер географических координат GPS из строк отчетов.
    Поддерживает:
      - '15 06.4N', '119 43.0 E' (градусы, минуты с точкой)
      - '09 24 7 N', '120 20 3 E' (градусы, минуты, десятые минут через пробел)
      - '1 04 9S', '139 51 6E'
      - '46.385 N' (десятичные градусы)
    """
    if not coord_str or not isinstance(coord_str, str):
        return None
    try:
        cleaned = coord_str.strip().upper().replace("°", " ").replace("'", " ").replace('"', " ")
        cleaned = re.sub(r'\.+', '.', cleaned)

        # 1. Формат из 3 чисел: '09 24 7 N' -> 9° 24.7' N
        m3 = re.search(r'(\d+)\s+(\d+)\s+([\d\.]+)\s*([NSEW])', cleaned)
        if m3:
            deg, m_int, m_dec, direction = m3.groups()
            clean_dec = re.sub(r'[^\d]', '', m_dec)
            minutes = float(f"{m_int}.{clean_dec}") if clean_dec else float(m_int)
            val = float(deg) + minutes / 60.0
            return -val if direction in ['S', 'W'] else val

        # 2. Формат из 2 чисел: '15 06.4N' -> 15° 6.4' N
        m2 = re.search(r'(\d+)\s+([\d\.]+)\s*([NSEW])', cleaned)
        if m2:
            deg, minutes, direction = m2.groups()
            min_match = re.search(r'\d+(?:\.\d+)?', minutes)
            if min_match:
                val = float(deg) + float(min_match.group(0)) / 60.0
                return -val if direction in ['S', 'W'] else val

        # 3. Десятичные градусы: '46.385 N'
        m1 = re.search(r'([\d\.]+)\s*([NSEW])', cleaned)
        if m1:
            deg, direction = m1.groups()
            deg_match = re.search(r'\d+(?:\.\d+)?', deg)
            if deg_match:
                val = float(deg_match.group(0))
                return -val if direction in ['S', 'W'] else val
    except Exception:
        return None
    return None


def find_first_number_in_row_after(sheet, row_idx: int, start_col_idx: int, max_cols: int = 12):
    """Ищет первое число в строке правее заданной колонки (с учетом объединенных ячеек)."""
    for col in range(start_col_idx + 1, min(start_col_idx + max_cols + 1, 26)):
        num = to_float(sheet.cell(row=row_idx, column=col).value)
        if num is not None:
            return num
    return None


def extract_sheet_data(sheet) -> dict:
    """
    Адаптивное извлечение данных суточного отчета из одного листа Excel:
    - Учитывает плавающие строки DRAFT (17, 18, 20, 21);
    - Находит скорость при отсутствии метки AV.SPEED (число между TRIP и LOG);
    - Распознает итоговый расход 'Total' и 'TOTAL: VLSFO';
    - Безопасно извлекает координаты GPS.
    """
    data = {
        "sog_source": None,
        "total_source": None
    }

    trip_cell = None
    log_cell = None
    draft_header_cell = None

    # Проход по ячейкам диапазона отчета (первые 45 строк, 16 колонок)
    for r in range(1, 45):
        for c in range(1, 16):
            cell_val = sheet.cell(row=r, column=c).value
            if cell_val is None:
                continue

            raw_text = str(cell_val).strip().upper()

            # 1. Расход Главного Двигателя (Main Engine, ME) в секции расхода (r >= 15)
            if ("MAIN ENGINE" in raw_text or "M/E" in raw_text) and r >= 15:
                if "me_consumption_tons" not in data:
                    num = find_first_number_in_row_after(sheet, r, c)
                    if num is not None and 0.0 <= num <= 100.0:
                        data["me_consumption_tons"] = num

            # 2. Вспомогательные генераторы (Auxiliary Generators, AE)
            elif ("AUXILIARY GENERATORS" in raw_text or "AUX GENERATORS" in raw_text or raw_text == "A/E") and r >= 15:
                if "ae_consumption_tons" not in data:
                    num = find_first_number_in_row_after(sheet, r, c)
                    if num is not None and 0.0 <= num <= 50.0:
                        data["ae_consumption_tons"] = num

            # 3. Суммарный расход топлива (Total VLSFO или Total)
            elif ("TOTAL: VLSFO" in raw_text or "TOTAL VLSFO" in raw_text or raw_text == "TOTAL") and r >= 20:
                if "total_vlsfo_tons" not in data:
                    num = find_first_number_in_row_after(sheet, r, c)
                    if num is not None and 0.0 <= num <= 150.0:
                        data["total_vlsfo_tons"] = num
                        data["total_source"] = raw_text

            # 4. Скорость судна (SOG) через текстовые метки
            elif "AV.SPEED" in raw_text or "AV SPEED" in raw_text or "AVERAGE SPEED" in raw_text:
                if "sog" not in data:
                    num = find_first_number_in_row_after(sheet, r, c)
                    if num is not None and 0.0 <= num <= 35.0:
                        data["sog"] = num
                        data["sog_source"] = "AV.SPEED"

            # Координаты ячеек TRIP и LOG для эвристики скорости без метки AV.SPEED
            elif raw_text == "TRIP" or "DAILY TRIP" in raw_text:
                trip_cell = (r, c)
            elif raw_text == "LOG" and "SPEED" not in raw_text:
                log_cell = (r, c)

            # 5. Ходовое время (STEAMING TIME)
            elif "STEAMING TIME" in raw_text or "STEAMING HRS" in raw_text:
                if "steaming_hours" not in data:
                    num = find_first_number_in_row_after(sheet, r, c)
                    if num is not None and 0.0 <= num <= 30.0:
                        data["steaming_hours"] = num

            # 6. Блок DRAFT (строка 17, 18, 20 или 21)
            elif raw_text == "DRAFT":
                draft_header_cell = (r, c)

            # 7. Осадки носом (FWD) и кормой (AFT)
            elif (raw_text in ["FWD", "FWD.", "FWD:"] or "DRAFT FWD" in raw_text) and "FW ROB" not in raw_text:
                if "draft_fwd" not in data:
                    num = find_first_number_in_row_after(sheet, r, c)
                    if num is not None and 3.0 <= num <= 25.0:
                        data["draft_fwd"] = num

            elif raw_text in ["AFT", "AFT.", "AFT:"] or "DRAFT AFT" in raw_text:
                if "draft_aft" not in data:
                    num = find_first_number_in_row_after(sheet, r, c)
                    if num is not None and 3.0 <= num <= 25.0:
                        data["draft_aft"] = num

            # 8. Координаты GPS (GPS POSN)
            elif "GPS POSN" in raw_text or raw_text == "GPS":
                for dr in range(0, 3):
                    for dc in range(-1, 5):
                        val_near = sheet.cell(row=r + dr, column=c + dc).value
                        if val_near and isinstance(val_near, str):
                            coord = parse_coordinate(val_near)
                            if coord is not None:
                                if ("N" in val_near.upper() or "S" in val_near.upper()) and "lat" not in data:
                                    data["lat"] = coord
                                elif ("E" in val_near.upper() or "W" in val_near.upper()) and "lon" not in data:
                                    data["lon"] = coord

    # --- Эвристика скорости 1: число между TRIP и LOG (типично для отчетов 2024 года) ---
    if "sog" not in data and trip_cell is not None:
        tr_r, tr_c = trip_cell
        max_r = log_cell[0] if (log_cell and log_cell[0] > tr_r) else tr_r + 2
        for cand_r in range(tr_r + 1, max_r + 1):
            for cand_c in range(tr_c, tr_c + 6):
                val = sheet.cell(row=cand_r, column=cand_c).value
                num = to_float(val)
                if num is not None and 0.5 <= num <= 30.0:
                    data["sog"] = num
                    data["sog_source"] = f"between_TRIP_LOG(R{cand_r}C{cand_c})"
                    break
            if "sog" in data:
                break

    # --- Эвристика скорости 2: расчетная скорость trip_distance / steaming_hours ---
    if "sog" not in data and trip_cell is not None:
        trip_dist = find_first_number_in_row_after(sheet, trip_cell[0], trip_cell[1])
        steaming = data.get("steaming_hours")
        if trip_dist is not None and steaming and steaming > 0:
            calc_sog = trip_dist / steaming
            if 0.5 <= calc_sog <= 30.0:
                data["sog"] = calc_sog
                data["sog_source"] = "trip_dist/steaming"

    # --- Эвристика осадки относительно ячейки DRAFT ---
    if draft_header_cell is not None and ("draft_fwd" not in data or "draft_aft" not in data):
        dr_r, dr_c = draft_header_cell
        for offset_r in range(1, 5):
            target_r = dr_r + offset_r
            if target_r > sheet.max_row:
                break
            cell_txt = str(sheet.cell(row=target_r, column=dr_c).value or "").strip().upper()
            if "FWD" in cell_txt and "draft_fwd" not in data:
                num = find_first_number_in_row_after(sheet, target_r, dr_c)
                if num is not None and 3.0 <= num <= 25.0:
                    data["draft_fwd"] = num
            elif "AFT" in cell_txt and "draft_aft" not in data:
                num = find_first_number_in_row_after(sheet, target_r, dr_c)
                if num is not None and 3.0 <= num <= 25.0:
                    data["draft_aft"] = num

    # Расчет средней осадки (mean_draft) и дифферента (trim)
    if data.get("draft_fwd") is not None and data.get("draft_aft") is not None:
        data["mean_draft"] = (data["draft_fwd"] + data["draft_aft"]) / 2.0
        data["trim"] = data["draft_aft"] - data["draft_fwd"]
    elif data.get("draft_fwd") is not None:
        data["mean_draft"] = data["draft_fwd"]
        data["trim"] = None
    elif data.get("draft_aft") is not None:
        data["mean_draft"] = data["draft_aft"]
        data["trim"] = None
    else:
        data["mean_draft"] = None
        data["trim"] = None

    # Удельный часовой расход главного двигателя (кг/ч)
    steaming = data.get("steaming_hours")
    me_tons = data.get("me_consumption_tons")
    if steaming and me_tons and steaming > 0:
        data["fuel_kg_per_h"] = (me_tons * 1000.0) / steaming
    else:
        data["fuel_kg_per_h"] = None

    return data


def main():
    base_dir = Path(__file__).resolve().parent
    excel_path = base_dir / "daily_reports.xlsx"
    output_path = base_dir / "processed_noon_dataset.csv"

    if not excel_path.exists():
        print(f"[ERROR] Файл '{excel_path}' не найден!")
        sys.exit(1)

    print(f"[INFO] Загрузка книги Excel: {excel_path}...")
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    all_sheet_names = wb.sheetnames
    total_sheets_count = len(all_sheet_names)
    print(f"[INFO] Всего листов в файле: {total_sheets_count}")

    parsed_rows = []
    skipped_sheets = []

    for idx, sname in enumerate(all_sheet_names, start=1):
        # Пропускаем вспомогательные листы расчетов (например, SLIP)
        if sname.strip().upper() == "SLIP":
            skipped_sheets.append((sname, "Вспомогательный лист (SLIP)"))
            continue

        sheet = wb[sname]
        try:
            row_data = extract_sheet_data(sheet)
            row_data["sheet_name"] = sname

            # Проверяем, есть ли минимальные данные суточного отчета
            if row_data.get("me_consumption_tons") is not None:
                parsed_rows.append(row_data)
            else:
                skipped_sheets.append((sname, "Расход Главного Двигателя не найден"))
        except Exception as e:
            skipped_sheets.append((sname, f"Ошибка парсинга: {e}"))

    wb.close()

    df = pd.DataFrame(parsed_rows)
    print("\n" + "=" * 65)
    print("[РЕЗУЛЬТАТЫ ПАРСИНГА СЫРЫХ ДАННЫХ]")
    print("=" * 65)
    print(f"-> Успешно обработано отчетов: {len(df)}")
    print(f"-> Пропущено не-Noon листов: {len(skipped_sheets)}")

    # Диагностика источников скорости и расхода
    print("\n[ИСТОЧНИКИ СКОРОСТИ SOG]")
    for src, cnt in df["sog_source"].value_counts(dropna=False).items():
        src_label = src if pd.notna(src) else "Не найдена (стоянка/якорь)"
        print(f"   - {src_label}: {cnt} листов")

    print("\n[ИСТОЧНИКИ СУММАРНОГО РАСХОДА ТОПЛИВА]")
    for src, cnt in df["total_source"].value_counts(dropna=False).items():
        src_label = src if pd.notna(src) else "Не найден"
        print(f"   - '{src_label}': {cnt} листов")

    print(f"\n[ЗАПОЛНЕННОСТЬ ПОЛЕЙ СЫРОГО ДАТАСЕТА]")
    print(f"   - SOG (скорость): {df['sog'].notna().sum()} / {len(df)}")
    print(f"   - ME расход: {df['me_consumption_tons'].notna().sum()} / {len(df)}")
    print(f"   - Steaming hours (ходовое время): {df['steaming_hours'].notna().sum()} / {len(df)}")
    print(f"   - Средняя осадка (mean_draft): {df['mean_draft'].notna().sum()} / {len(df)}")
    print(f"   - GPS координаты (lat/lon): {df['lat'].notna().sum()} / {len(df)}")

    # =========================================================================
    # ФИЛЬТРАЦИЯ ДЛЯ КРЕЙСЕРСКОГО ХОДА В МОРЕ
    # =========================================================================
    # 1. sog >= 8.0 узлов (отсекаем маневры, швартовки, стоянки, дрейф)
    # 2. steaming_hours >= 18.0 часов (полноценные ходовые сутки)
    # 3. me_consumption_tons >= 8.0 тонн (чистый ход под нагрузкой ГД)
    # =========================================================================
    mask_cruise = (
        (df["sog"] >= 8.0) &
        (df["steaming_hours"] >= 18.0) &
        (df["me_consumption_tons"] >= 8.0)
    )
    df_cruise = df[mask_cruise].copy()

    print("\n" + "=" * 65)
    print("[ФИЛЬТРАЦИЯ КРЕЙСЕРСКОГО ХОДА В МОРЕ]")
    print("   Критерии: sog >= 8.0 уз, steaming_hours >= 18.0 ч, me_consumption_tons >= 8.0 т")
    print("=" * 65)
    print(f"-> Отобрано чистых ходовых суток: {len(df_cruise)}")

    # Анализ пропусков в крейсерском режиме
    nan_sog = df_cruise[df_cruise["sog"].isna()]
    nan_draft = df_cruise[df_cruise["mean_draft"].isna()]

    print(f"-> Крейсерских листов с пропуском SOG: {len(nan_sog)}")
    print(f"-> Крейсерских листов с пропуском mean_draft: {len(nan_draft)}")

    if len(nan_draft) > 0:
        print("\n[СПИСОК ЛИСТОВ КРЕЙСЕРСКОГО ХОДА С ПРОПУСКОМ ОСАДКИ (NaN in mean_draft)]")
        print("-" * 65)
        for _, r in nan_draft.iterrows():
            print(f"   * Лист '{r['sheet_name']}': SOG={r['sog']:.2f} уз, "
                  f"Время={r['steaming_hours']} ч, ME={r['me_consumption_tons']} т "
                  f"[ячейки DRAFT пусты в Excel-отчете]")
        print("-" * 65)

    # Итоговые колонки для сохранения
    output_columns = [
        "lat",
        "lon",
        "steaming_hours",
        "me_consumption_tons",
        "ae_consumption_tons",
        "fuel_kg_per_h",
        "sheet_name",
        "draft_fwd",
        "draft_aft",
        "mean_draft",
        "trim",
        "sog",
        "total_vlsfo_tons"
    ]

    df_final = df_cruise[output_columns].copy()
    df_final.to_csv(output_path, index=False)
    print(f"\n[OK] Датасет успешно сохранен в '{output_path}'")
    print(f"[OK] Размерность итогового датасета: {df_final.shape[0]} строк x {df_final.shape[1]} колонок")

    print("\n[ОСНОВНЫЕ СТАТИСТИКИ КРЕЙСЕРСКОГО ХОДА]")
    print(df_final[["sog", "steaming_hours", "me_consumption_tons", "mean_draft", "fuel_kg_per_h"]].describe().to_string())


if __name__ == "__main__":
    main()