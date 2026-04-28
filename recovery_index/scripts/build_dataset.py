from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT_DIR / "recovery_index"
BASE_SEED_FILE = PROJECT_DIR / "data" / "companies_seed.csv"
RAW_RUN_DIR = PROJECT_DIR / "service" / "raw"
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_FILE = DATA_DIR / "model_dataset.csv"
OBSERVATION_END = pd.Timestamp("2026-04-23")

FINANCIAL_LINES = {
    "1100": "noncurrent_assets",
    "1150": "fixed_assets",
    "1200": "current_assets",
    "1210": "inventory",
    "1230": "receivables",
    "1250": "cash",
    "1300": "equity",
    "1500": "liabilities",
    "1520": "payables",
    "1600": "assets",
    "1700": "balance",
    "2100": "gross_profit",
    "2110": "revenue",
    "2120": "cost",
    "2200": "operating_profit",
    "2300": "pretax_profit",
    "2340": "other_income",
    "2400": "net_profit",
    "2500": "result",
}

BANKRUPTCY_KEYWORDS = (
    "банкрот",
    "конкурс",
    "наблюдени",
    "внешнее управление",
    "финансовое оздоровление",
    "несостоятель",
)

EFRSB_CATEGORY_PATTERNS = {
    "meeting": ("собрании кредиторов", "результатах проведения собрания кредиторов"),
    "creditor_demand": ("получении требования кредитора",),
    "court_act": ("судебном акте",),
    "trades": ("торгов", "договора купли-продажи"),
    "inventory": ("инвентаризации имущества", "оценке имущества"),
    "contest_transaction": ("сделки должника",),
    "subsidiary": ("субсидиарной ответственности", "контролирующих должника"),
    "false_bankruptcy": ("преднамеренного или фиктивного банкротства",),
    "statement_acceptance": ("принятии заявления о признании должника банкротом",),
    "workers": ("работников",),
}

def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def company_raw_dir(company_inn: str) -> Path:
    return RAW_RUN_DIR / company_inn

def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number

def month_diff(left: pd.Timestamp, right: pd.Timestamp) -> int:
    return (right.year - left.year) * 12 + (right.month - left.month)

def bool_int(value: Any) -> int:
    return int(bool(value))

def classify_efrsb_type(message_type: str) -> str:
    lowered = (message_type or "").lower()
    for category, patterns in EFRSB_CATEGORY_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            return category
    return "other"

def build_case_frame(company_inn: str) -> pd.DataFrame:
    payload = read_json(company_raw_dir(company_inn) / "legal_cases_pages.json")
    pages = payload.get("pages") or []
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    for page in pages:
        items = (page.get("data") or {}).get("Записи") or []
        for item in items:
            case_uuid = item.get("UUID")
            case_number = item.get("Номер")
            case_date = pd.to_datetime(item.get("Дата"), errors="coerce")
            unique_key = (case_uuid or case_number, str(case_date))
            if unique_key in seen_keys:
                continue
            seen_keys.add(unique_key)

            plaintiffs = item.get("Ист") or []
            defendants = item.get("Ответ") or []
            in_plaintiffs = any(str(entry.get("ИНН") or "") == company_inn for entry in plaintiffs)
            in_defendants = any(str(entry.get("ИНН") or "") == company_inn for entry in defendants)
            if in_plaintiffs and in_defendants:
                role = "both"
            elif in_defendants:
                role = "defendant"
            elif in_plaintiffs:
                role = "plaintiff"
            else:
                role = "other"

            rows.append(
                {
                    "company_inn": company_inn,
                    "case_uuid": case_uuid or case_number,
                    "case_number": case_number,
                    "case_date": case_date,
                    "case_month": case_date.to_period("M").to_timestamp() if pd.notna(case_date) else pd.NaT,
                    "claim_amount": safe_float(item.get("СуммИск")) or 0.0,
                    "role": role,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["company_inn", "case_uuid", "case_number", "case_date", "case_month", "claim_amount", "role"]
        )
    frame = pd.DataFrame(rows).sort_values("case_date").reset_index(drop=True)
    return frame

def build_enforcement_frame(company_inn: str) -> pd.DataFrame:
    payload = read_json(company_raw_dir(company_inn) / "enforcements_pages.json")
    pages = payload.get("pages") or []
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    for page in pages:
        items = (page.get("data") or {}).get("Записи") or []
        for item in items:
            number = item.get("ИспПрНомер")
            date = pd.to_datetime(item.get("ИспПрДата"), errors="coerce")
            unique_key = (number, str(date))
            if unique_key in seen_keys:
                continue
            seen_keys.add(unique_key)
            rows.append(
                {
                    "company_inn": company_inn,
                    "enforcement_number": number,
                    "enforcement_date": date,
                    "debt_sum": safe_float(item.get("СумДолг")) or 0.0,
                    "remaining_debt_sum": safe_float(item.get("ОстЗадолж")) or 0.0,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["company_inn", "enforcement_number", "enforcement_date", "debt_sum", "remaining_debt_sum"])
    return pd.DataFrame(rows).sort_values("enforcement_date").reset_index(drop=True)

def build_efrsb_frame(company_inn: str) -> pd.DataFrame:
    payload = read_json(company_raw_dir(company_inn) / "company.json")
    data = payload.get("data") or {}
    rows: list[dict[str, Any]] = []
    for item in data.get("ЕФРСБ") or []:
        message_type = item.get("Тип") or ""
        date = pd.to_datetime(item.get("Дата"), errors="coerce")
        rows.append(
            {
                "company_inn": company_inn,
                "message_date": date,
                "message_type": message_type,
                "message_category": classify_efrsb_type(message_type),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["company_inn", "message_date", "message_type", "message_category"])
    return pd.DataFrame(rows).sort_values("message_date").reset_index(drop=True)

def parse_financial_years(company_inn: str) -> dict[int, dict[str, float]]:
    payload = read_json(company_raw_dir(company_inn) / "finances.json")
    data = payload.get("data") or {}
    result: dict[int, dict[str, float]] = {}
    for year_key, year_payload in data.items():
        if not isinstance(year_payload, dict):
            continue
        try:
            year = int(year_key)
        except ValueError:
            continue
        line_values: dict[str, float] = {}
        for line_code in FINANCIAL_LINES:
            value = safe_float(year_payload.get(line_code))
            if value is not None:
                line_values[line_code] = value
        if line_values:
            result[year] = line_values
    return result

def snapshot_features(financial_years: dict[int, dict[str, float]], anchor_date: pd.Timestamp, prefix: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if pd.isna(anchor_date):
        output[f"{prefix}_snapshot_year"] = np.nan
        output[f"{prefix}_has_snapshot"] = 0
        return output

    cutoff_year = anchor_date.year - 1
    available_years = sorted(year for year in financial_years if year <= cutoff_year)
    if not available_years:
        output[f"{prefix}_snapshot_year"] = np.nan
        output[f"{prefix}_has_snapshot"] = 0
        return output

    year0 = available_years[-1]
    year1 = available_years[-2] if len(available_years) >= 2 else None
    year2 = available_years[-3] if len(available_years) >= 3 else None
    output[f"{prefix}_snapshot_year"] = year0
    output[f"{prefix}_has_snapshot"] = 1

    for line_code, line_name in FINANCIAL_LINES.items():
        v0 = financial_years.get(year0, {}).get(line_code)
        v1 = financial_years.get(year1, {}).get(line_code) if year1 is not None else None
        v2 = financial_years.get(year2, {}).get(line_code) if year2 is not None else None
        output[f"{prefix}_{line_name}_t0"] = v0
        output[f"{prefix}_{line_name}_t1"] = v1
        output[f"{prefix}_{line_name}_t2"] = v2
        output[f"{prefix}_{line_name}_delta1"] = (v0 - v1) if v0 is not None and v1 is not None else np.nan
        output[f"{prefix}_{line_name}_delta2"] = (v1 - v2) if v1 is not None and v2 is not None else np.nan
        output[f"{prefix}_{line_name}_growth1"] = (
            ((v0 - v1) / abs(v1)) if v0 is not None and v1 not in (None, 0) else np.nan
        )

    assets = output.get(f"{prefix}_assets_t0")
    liabilities = output.get(f"{prefix}_liabilities_t0")
    equity = output.get(f"{prefix}_equity_t0")
    revenue = output.get(f"{prefix}_revenue_t0")
    profit = output.get(f"{prefix}_net_profit_t0")
    cash = output.get(f"{prefix}_cash_t0")
    payables = output.get(f"{prefix}_payables_t0")
    receivables = output.get(f"{prefix}_receivables_t0")

    output[f"{prefix}_liabilities_to_assets"] = (liabilities / assets) if assets not in (None, 0) and liabilities is not None else np.nan
    output[f"{prefix}_equity_to_assets"] = (equity / assets) if assets not in (None, 0) and equity is not None else np.nan
    output[f"{prefix}_profit_margin"] = (profit / revenue) if revenue not in (None, 0) and profit is not None else np.nan
    output[f"{prefix}_cash_to_assets"] = (cash / assets) if assets not in (None, 0) and cash is not None else np.nan
    output[f"{prefix}_receivables_to_assets"] = (receivables / assets) if assets not in (None, 0) and receivables is not None else np.nan
    output[f"{prefix}_payables_to_liabilities"] = (
        (payables / liabilities) if liabilities not in (None, 0) and payables is not None else np.nan
    )
    return output

def aggregate_case_window(case_frame: pd.DataFrame, anchor_date: pd.Timestamp, months: int, prefix: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if case_frame.empty or pd.isna(anchor_date):
        for field in (
            "count",
            "claim_sum",
            "claim_max",
            "defendant_count",
            "plaintiff_count",
            "defendant_claim_sum",
            "plaintiff_claim_sum",
            "active_months",
        ):
            output[f"{prefix}_{field}"] = 0.0
        return output

    start_date = anchor_date - pd.DateOffset(months=months)
    subset = case_frame[(case_frame["case_date"] <= anchor_date) & (case_frame["case_date"] > start_date)].copy()
    output[f"{prefix}_count"] = float(len(subset))
    output[f"{prefix}_claim_sum"] = float(subset["claim_amount"].sum()) if not subset.empty else 0.0
    output[f"{prefix}_claim_max"] = float(subset["claim_amount"].max()) if not subset.empty else 0.0
    output[f"{prefix}_defendant_count"] = float((subset["role"] == "defendant").sum())
    output[f"{prefix}_plaintiff_count"] = float((subset["role"] == "plaintiff").sum())
    output[f"{prefix}_defendant_claim_sum"] = float(subset.loc[subset["role"] == "defendant", "claim_amount"].sum())
    output[f"{prefix}_plaintiff_claim_sum"] = float(subset.loc[subset["role"] == "plaintiff", "claim_amount"].sum())
    output[f"{prefix}_active_months"] = float(subset["case_month"].nunique()) if not subset.empty else 0.0
    return output

def aggregate_previous_case_window(case_frame: pd.DataFrame, anchor_date: pd.Timestamp, months: int, prefix: str) -> dict[str, Any]:
    if pd.isna(anchor_date):
        return aggregate_case_window(case_frame, anchor_date, months, prefix)
    previous_anchor = pd.Timestamp(anchor_date) - pd.DateOffset(months=months)
    previous_frame = case_frame[case_frame["case_date"] <= previous_anchor].copy()
    return aggregate_case_window(previous_frame, previous_anchor, months, prefix)

def case_window_comparison_features(row: dict[str, Any], current_prefix: str, previous_prefix: str, output_prefix: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in (
        "count",
        "claim_sum",
        "claim_max",
        "defendant_count",
        "plaintiff_count",
        "defendant_claim_sum",
        "plaintiff_claim_sum",
        "active_months",
    ):
        current = safe_float(row.get(f"{current_prefix}_{field}")) or 0.0
        previous = safe_float(row.get(f"{previous_prefix}_{field}")) or 0.0
        output[f"{output_prefix}_{field}_diff"] = current - previous
        output[f"{output_prefix}_{field}_log_change"] = float(np.log1p(current) - np.log1p(previous))
    return output

def defendant_burst_strength(burst: dict[str, Any]) -> tuple[float, float]:
    return (
        float(burst.get("defendant_claim_sum") or 0.0),
        float(burst.get("defendant_cases") or 0.0),
    )

def build_bursts(case_frame: pd.DataFrame) -> list[dict[str, Any]]:
    if case_frame.empty:
        return []
    monthly = (
        case_frame.groupby("case_month", as_index=False)
        .agg(
            total_cases=("case_uuid", "nunique"),
            claim_sum=("claim_amount", "sum"),
            claim_max=("claim_amount", "max"),
            defendant_cases=("role", lambda s: int((s == "defendant").sum())),
            plaintiff_cases=("role", lambda s: int((s == "plaintiff").sum())),
            defendant_claim_sum=("claim_amount", lambda s: float(case_frame.loc[s.index][case_frame.loc[s.index, "role"] == "defendant"]["claim_amount"].sum())),
            plaintiff_claim_sum=("claim_amount", lambda s: float(case_frame.loc[s.index][case_frame.loc[s.index, "role"] == "plaintiff"]["claim_amount"].sum())),
        )
        .sort_values("case_month")
        .reset_index(drop=True)
    )

    bursts: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in monthly.to_dict("records"):
        month = row["case_month"]
        if not current:
            current = [row]
            continue
        prev = current[-1]["case_month"]
        if month_diff(prev, month) <= 2:
            current.append(row)
        else:
            bursts.append(current)
            current = [row]
    if current:
        bursts.append(current)

    monthly_lookup = monthly.set_index("case_month")
    output: list[dict[str, Any]] = []
    for chunk in bursts:
        start_month = chunk[0]["case_month"]
        end_month = chunk[-1]["case_month"]
        burst_months = month_diff(start_month, end_month) + 1
        prev_start = start_month - pd.DateOffset(months=12)
        prev_months = monthly_lookup[(monthly_lookup.index < start_month) & (monthly_lookup.index >= prev_start)]
        prev_case_mean = float(prev_months["total_cases"].mean()) if not prev_months.empty else 0.0
        prev_claim_mean = float(prev_months["claim_sum"].mean()) if not prev_months.empty else 0.0

        total_cases = float(sum(item["total_cases"] for item in chunk))
        total_claim_sum = float(sum(item["claim_sum"] for item in chunk))
        peak_cases = float(max(item["total_cases"] for item in chunk))
        peak_claim_sum = float(max(item["claim_sum"] for item in chunk))
        peak_claim_max = float(max(item["claim_max"] for item in chunk))
        last_month_cases = float(chunk[-1]["total_cases"])
        last_month_claim_sum = float(chunk[-1]["claim_sum"])
        defendant_cases = float(sum(item["defendant_cases"] for item in chunk))
        plaintiff_cases = float(sum(item["plaintiff_cases"] for item in chunk))
        defendant_claim_sum = float(sum(item["defendant_claim_sum"] for item in chunk))
        plaintiff_claim_sum = float(sum(item["plaintiff_claim_sum"] for item in chunk))

        output.append(
            {
                "start_date": start_month,
                "end_date": end_month,
                "months": float(burst_months),
                "total_cases": total_cases,
                "total_claim_sum": total_claim_sum,
                "peak_cases": peak_cases,
                "peak_claim_sum": peak_claim_sum,
                "peak_claim_max": peak_claim_max,
                "last_month_cases": last_month_cases,
                "last_month_claim_sum": last_month_claim_sum,
                "case_ratio": total_cases / max(prev_case_mean * burst_months, 1.0),
                "claim_ratio": total_claim_sum / max(prev_claim_mean * burst_months, 1.0),
                "cases_per_month": total_cases / max(burst_months, 1.0),
                "claim_per_month": total_claim_sum / max(burst_months, 1.0),
                "avg_claim_per_case": total_claim_sum / max(total_cases, 1.0),
                "peak_claim_share": peak_claim_max / max(total_claim_sum, 1.0),
                "defendant_cases": defendant_cases,
                "plaintiff_cases": plaintiff_cases,
                "defendant_claim_sum": defendant_claim_sum,
                "plaintiff_claim_sum": plaintiff_claim_sum,
            }
        )
    return output

def burst_features(burst: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    fields = [
        "start_date",
        "end_date",
        "months",
        "total_cases",
        "total_claim_sum",
        "peak_cases",
        "peak_claim_sum",
        "peak_claim_max",
        "last_month_cases",
        "last_month_claim_sum",
        "case_ratio",
        "claim_ratio",
        "cases_per_month",
        "claim_per_month",
        "avg_claim_per_case",
        "peak_claim_share",
        "defendant_cases",
        "plaintiff_cases",
        "defendant_claim_sum",
        "plaintiff_claim_sum",
    ]
    if burst is None:
        return {f"{prefix}_{field}": (pd.NaT if field.endswith("date") else np.nan) for field in fields} | {f"{prefix}_has_burst": 0}
    output = {f"{prefix}_{field}": burst.get(field) for field in fields}
    output[f"{prefix}_has_burst"] = 1
    return output

def enforcement_features(enforcement_frame: pd.DataFrame, anchor_date: pd.Timestamp, prefix: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if enforcement_frame.empty or pd.isna(anchor_date):
        for field in (
            "count_total",
            "debt_total",
            "remaining_total",
            "debt_max",
            "count_12m",
            "debt_12m",
            "count_18m",
            "debt_18m",
            "has_any",
            "days_since_last",
        ):
            output[f"{prefix}_{field}"] = 0.0 if field != "days_since_last" else np.nan
        return output

    subset = enforcement_frame[enforcement_frame["enforcement_date"] <= anchor_date].copy()
    last_12m = subset[subset["enforcement_date"] > anchor_date - pd.DateOffset(months=12)]
    last_18m = subset[subset["enforcement_date"] > anchor_date - pd.DateOffset(months=18)]
    output[f"{prefix}_count_total"] = float(len(subset))
    output[f"{prefix}_debt_total"] = float(subset["debt_sum"].sum()) if not subset.empty else 0.0
    output[f"{prefix}_remaining_total"] = float(subset["remaining_debt_sum"].sum()) if not subset.empty else 0.0
    output[f"{prefix}_debt_max"] = float(subset["debt_sum"].max()) if not subset.empty else 0.0
    output[f"{prefix}_count_12m"] = float(len(last_12m))
    output[f"{prefix}_debt_12m"] = float(last_12m["debt_sum"].sum()) if not last_12m.empty else 0.0
    output[f"{prefix}_count_18m"] = float(len(last_18m))
    output[f"{prefix}_debt_18m"] = float(last_18m["debt_sum"].sum()) if not last_18m.empty else 0.0
    output[f"{prefix}_has_any"] = float(not subset.empty)
    output[f"{prefix}_days_since_last"] = (
        float((anchor_date - subset["enforcement_date"].max()).days) if not subset.empty else np.nan
    )
    return output

def efrsb_features(efrsb_frame: pd.DataFrame, anchor_date: pd.Timestamp, prefix: str) -> dict[str, Any]:
    categories = sorted(set(EFRSB_CATEGORY_PATTERNS) | {"other"})
    output: dict[str, Any] = {f"{prefix}_total_count": 0.0, f"{prefix}_active_months": 0.0}
    for category in categories:
        output[f"{prefix}_{category}_count"] = 0.0
        output[f"{prefix}_{category}_count_12m"] = 0.0
        output[f"{prefix}_{category}_count_18m"] = 0.0
    if efrsb_frame.empty or pd.isna(anchor_date):
        output[f"{prefix}_days_since_last"] = np.nan
        output[f"{prefix}_has_any"] = 0.0
        return output

    subset = efrsb_frame[efrsb_frame["message_date"] <= anchor_date].copy()
    last_12m = subset[subset["message_date"] > anchor_date - pd.DateOffset(months=12)]
    last_18m = subset[subset["message_date"] > anchor_date - pd.DateOffset(months=18)]
    output[f"{prefix}_total_count"] = float(len(subset))
    output[f"{prefix}_active_months"] = float(
        subset["message_date"].dt.to_period("M").nunique()
    ) if not subset.empty else 0.0
    for category in categories:
        output[f"{prefix}_{category}_count"] = float((subset["message_category"] == category).sum())
        output[f"{prefix}_{category}_count_12m"] = float((last_12m["message_category"] == category).sum())
        output[f"{prefix}_{category}_count_18m"] = float((last_18m["message_category"] == category).sum())
    output[f"{prefix}_days_since_last"] = float((anchor_date - subset["message_date"].max()).days) if not subset.empty else np.nan
    output[f"{prefix}_has_any"] = float(not subset.empty)
    return output

def raw_profile_features(company_inn: str) -> dict[str, Any]:
    payload = read_json(company_raw_dir(company_inn) / "company.json")
    data = payload.get("data") or {}
    contacts = data.get("Контакты") or {}
    address = data.get("ЮрАдрес") or {}
    subdivisions = data.get("Подразд") or {}
    founders = data.get("Учред") or {}
    output = {
        "raw_has_contacts": bool_int(bool(contacts)),
        "raw_phone_count": float(len(contacts.get("Тел") or [])),
        "raw_email_count": float(len(contacts.get("Емэйл") or [])),
        "raw_has_website": bool_int(bool(contacts.get("ВебСайт"))),
        "raw_licenses_count": float(len(data.get("Лиценз") or [])),
        "raw_support_count": float(len(data.get("ПоддержМСП") or [])),
        "raw_trademark_count": float(len(data.get("ТоварЗнак") or [])),
        "raw_branch_count": float(len((subdivisions.get("Филиал") or []))),
        "raw_rep_office_count": float(len((subdivisions.get("Представ") or []))),
        "raw_predecessor_count": float(len(data.get("Правопредш") or [])),
        "raw_successor_count": float(len(data.get("Правопреем") or [])),
        "raw_manager_count": float(len(data.get("Руковод") or [])),
        "raw_founder_fl_count": float(len(founders.get("ФЛ") or [])),
        "raw_founder_rol_count": float(len(founders.get("РосОрг") or [])),
        "raw_founder_foreign_count": float(len(founders.get("ИнОрг") or [])),
        "raw_founder_public_count": float(len(founders.get("РФ") or [])),
        "raw_has_registry_holder": bool_int(bool(data.get("ДержРеестрАО"))),
        "raw_address_unreliable_flag": bool_int(address.get("Недост")),
        "raw_mass_address_count": float(len(address.get("МассАдрес") or [])),
        "raw_sanctions_country_count": float(len(data.get("СанкцииСтраны") or [])),
        "raw_negative_supplier_flag": bool_int(data.get("НедобПост")),
        "raw_illegal_finance_flag": bool_int(data.get("НелегалФин")),
        "raw_sanctions_flag": bool_int(data.get("Санкции")),
    }
    status_name = ((data.get("Статус") or {}).get("Наим") or "").lower()
    liquid_reason = ((data.get("Ликвид") or {}).get("Наим") or "").lower()
    output["bankruptcy_final_label"] = int(any(keyword in status_name for keyword in BANKRUPTCY_KEYWORDS) or any(keyword in liquid_reason for keyword in BANKRUPTCY_KEYWORDS))
    return output

def company_profile_from_raw(company_inn: str) -> pd.Series:
    payload = read_json(company_raw_dir(company_inn) / "company.json")
    data = payload.get("data") or {}
    status = data.get("Статус") or {}
    liquid = data.get("Ликвид") or {}
    capital = data.get("УстКап") or {}
    taxes = data.get("Налоги") or {}
    return pd.Series(
        {
            "company_inn": company_inn,
            "status_name": status.get("Наим"),
            "status_code": status.get("Код"),
            "liquid_date": liquid.get("Дата"),
            "liquid_reason": liquid.get("Наим"),
            "region_code_api": (data.get("Регион") or {}).get("Код"),
            "okved_code_api": (data.get("ОКВЭД") or {}).get("Код"),
            "okved_name_api": (data.get("ОКВЭД") or {}).get("Наим"),
            "capital_sum_api": safe_float(capital.get("Сумма")),
            "headcount_api": safe_float(data.get("СЧР")),
            "tax_paid_sum_api": safe_float(taxes.get("СумУпл")),
            "tax_debt_sum_api": safe_float(taxes.get("СумНедоим")),
            "msp_category_api": (data.get("РМСП") or {}).get("Кат"),
            "sanctions_flag_api": data.get("Санкции"),
            "mass_director_flag_api": data.get("МассРуковод"),
            "mass_founder_flag_api": data.get("МассУчред"),
            "disqualified_persons_flag_api": data.get("ДисквЛица"),
        }
    )

def prefixed_profile(profile_row: pd.Series | None) -> dict[str, Any]:
    if profile_row is None:
        return {}
    output = {}
    for source, target in (
        ("region_code_api", "profile_region_code"),
        ("okved_code_api", "profile_okved_code"),
        ("okved_name_api", "profile_okved_name"),
        ("capital_sum_api", "profile_capital_sum"),
        ("headcount_api", "profile_headcount"),
        ("tax_paid_sum_api", "profile_tax_paid_sum"),
        ("tax_debt_sum_api", "profile_tax_debt_sum"),
        ("msp_category_api", "profile_msp_category"),
        ("sanctions_flag_api", "profile_sanctions_flag"),
        ("mass_director_flag_api", "profile_mass_director_flag"),
        ("mass_founder_flag_api", "profile_mass_founder_flag"),
        ("disqualified_persons_flag_api", "profile_disqualified_persons_flag"),
    ):
        output[target] = profile_row.get(source)
    output["profile_tax_debt_to_tax_paid_ratio"] = (
        output["profile_tax_debt_sum"] / output["profile_tax_paid_sum"]
        if output.get("profile_tax_paid_sum") not in (None, 0, np.nan) and pd.notna(output.get("profile_tax_paid_sum")) and pd.notna(output.get("profile_tax_debt_sum"))
        else np.nan
    )
    return output

def burst_target(outcome_date: pd.Timestamp, anchor_date: pd.Timestamp | Any, months: int) -> int:
    if pd.isna(outcome_date) or pd.isna(anchor_date):
        return 0
    distance_months = (outcome_date - pd.Timestamp(anchor_date)).days / 30.44
    return int(distance_months >= 0 and distance_months <= months)

def build_rows() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(BASE_SEED_FILE, dtype={"company_inn": "string", "company_ogrn": "string"})

    rows: list[dict[str, Any]] = []
    for sample_row in sample.to_dict("records"):
        company_inn = str(sample_row["company_inn"])
        outcome_date = pd.to_datetime(sample_row.get("first_negative_outcome_date"), errors="coerce")
        final_cutoff = outcome_date - pd.Timedelta(days=1) if pd.notna(outcome_date) else OBSERVATION_END

        company_profile = company_profile_from_raw(company_inn)
        raw_profile = raw_profile_features(company_inn)
        case_frame = build_case_frame(company_inn)
        enforcement_frame = build_enforcement_frame(company_inn)
        efrsb_frame = build_efrsb_frame(company_inn)
        finances = parse_financial_years(company_inn)

        case_pre_cutoff = case_frame[case_frame["case_date"] <= final_cutoff].copy()
        bursts = build_bursts(case_pre_cutoff)
        last_burst = bursts[-1] if bursts else None
        maxsafe_burst = max(bursts, key=defendant_burst_strength, default=None)
        last_defendant_claim = safe_float(last_burst.get("defendant_claim_sum") if last_burst else None) or 0.0
        maxsafe_defendant_claim = safe_float(maxsafe_burst.get("defendant_claim_sum") if maxsafe_burst else None) or 0.0

        row: dict[str, Any] = {
            **sample_row,
            "failed_label": int(sample_row["final_status_label"] == "failed"),
            **prefixed_profile(company_profile),
            **raw_profile,
            **aggregate_case_window(case_pre_cutoff, final_cutoff, 12, "final_window12"),
            **aggregate_case_window(case_pre_cutoff, final_cutoff, 18, "final_window18"),
            **aggregate_case_window(case_pre_cutoff, final_cutoff, 24, "final_window24"),
            **aggregate_previous_case_window(case_pre_cutoff, final_cutoff, 12, "final_prev12"),
            **aggregate_case_window(case_pre_cutoff, final_cutoff, 999, "final_alltime"),
            **snapshot_features(finances, final_cutoff, "final_fin"),
            **enforcement_features(enforcement_frame, final_cutoff, "final_enf"),
            **efrsb_features(efrsb_frame, final_cutoff, "final_efrsb"),
            **burst_features(last_burst, "last_burst"),
            **burst_features(maxsafe_burst, "maxsafe"),
            "safe_history_max_to_last_defendant_claim_ratio": maxsafe_defendant_claim / max(last_defendant_claim, 1.0),
        }
        row |= case_window_comparison_features(row, "final_window12", "final_prev12", "final_window12_vs_prev12")

        for name, burst in (
            ("last", last_burst),
            ("maxsafe", maxsafe_burst),
        ):
            anchor_date = pd.Timestamp(burst["end_date"]) if burst is not None else pd.NaT
            if name == "last":
                window_prefix = "last_anchor"
            else:
                window_prefix = "maxsafe"
            row[f"failed_within_12m_from_{name}"] = burst_target(outcome_date, anchor_date, 12) if row["failed_label"] else 0
            row[f"failed_within_18m_from_{name}"] = burst_target(outcome_date, anchor_date, 18) if row["failed_label"] else 0
            row[f"bankruptcy_or_failed_within_12m_from_{name}"] = int(raw_profile["bankruptcy_final_label"] or row[f"failed_within_12m_from_{name}"])
            row[f"bankruptcy_or_failed_within_18m_from_{name}"] = int(raw_profile["bankruptcy_final_label"] or row[f"failed_within_18m_from_{name}"])
            row |= snapshot_features(finances, anchor_date, f"{name}_anchor_fin")
            row |= enforcement_features(enforcement_frame, anchor_date, f"{name}_anchor_enf")
            row |= efrsb_features(efrsb_frame, anchor_date, f"{name}_anchor_efrsb")
            row |= aggregate_case_window(case_frame[case_frame["case_date"] <= anchor_date], anchor_date, 12, f"{window_prefix}_window12")
            row |= aggregate_case_window(case_frame[case_frame["case_date"] <= anchor_date], anchor_date, 18, f"{window_prefix}_window18")
            row |= aggregate_case_window(case_frame[case_frame["case_date"] <= anchor_date], anchor_date, 24, f"{window_prefix}_window24")
            row |= aggregate_previous_case_window(case_frame[case_frame["case_date"] <= anchor_date], anchor_date, 12, f"{window_prefix}_prev12")
            row |= case_window_comparison_features(row, f"{window_prefix}_window12", f"{window_prefix}_prev12", f"{window_prefix}_window12_vs_prev12")

        row["anchor_variant"] = "last_burst_anchor"
        row["failed_within_12m_from_anchor"] = row["failed_within_12m_from_last"]

        rows.append(row)

    frame = pd.DataFrame(rows)
    return frame

def main() -> None:
    frame = build_rows()
    frame.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
