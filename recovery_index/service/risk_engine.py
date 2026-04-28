from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT_DIR / "recovery_index"
SERVICE_DIR = PROJECT_DIR / "service"
RAW_DIR = SERVICE_DIR / "raw"
REPORTS_DIR = SERVICE_DIR / "reports"
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
MODEL_FILE = MODELS_DIR / "risk_model_12m.joblib"
SERVICE_SUMMARY_FILE = MODELS_DIR / "risk_model_summary.json"
SERVICE_FEATURES_FILE = MODELS_DIR / "risk_model_features.json"
OBSERVATION_END = pd.Timestamp("2026-04-23")
DEFAULT_COURT_LOOKBACK_YEARS = 3
DEFAULT_MAX_EXTRA_CASE_PAGES = 200
DEFAULT_MAX_EXTRA_ENFORCEMENT_PAGES = 1
MAX_FINANCIAL_SNAPSHOT_AGE_YEARS = 3
CBR_INSURANCE_REPORT_YEAR = 2024
CBR_INSURANCE_RATIO_DATE = "31.12.2024"
CBR_INSURANCE_CARD_URL = "https://www.cbr.ru/finorg/foinfo/?ogrn={ogrn}"
CBR_INSURANCE_RATIO_URL = f"https://www.cbr.ru/insurance/standard_ratio_funds/standard_ratio_funds_data?DY={CBR_INSURANCE_RATIO_DATE}"
CBR_INSURANCE_PERFORMANCE_PAGE_URL = "https://cbr.ru/statistics/insurance/performance_indicators_ins/2024_4/"
CBR_INSURANCE_PERFORMANCE_ZIP_URL = "https://cbr.ru/Content/Document/File/174435/2024_4.zip"
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 recovery-index-prototype"}

sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from api_client import get_json
import build_dataset as bem

NEGATIVE_STATUS_KEYWORDS = (
    "не действует",
    "ликвид",
    "банкрот",
    "конкурс",
    "несостоятель",
    "прекращ",
    "исключ",
    "наблюдение",
    "внешнее управление",
)

RISK_LEVEL_RU = {
    "high": "высокий",
    "watch": "пограничный",
    "medium": "умеренный",
    "low": "низкий",
    "not_scored": "не рассчитывался",
}

GATE_MODE_RU = {
    "predictive": "прогноз",
    "known_negative": "уже негативный статус",
    "status_uncertain": "неопределенный статус",
}

TARGET_RU = {
    "failed_within_12m_from_last": "негативный исход в течение 12 месяцев после последнего судебного всплеска",
    "failed_within_12m_from_anchor": "негативный исход в течение 12 месяцев от даты оценки",
}

BASELINE_SCOPE_RU = {
    "industry": "сравнение по отрасли",
    "global_fallback": "сравнение по всей выборке, потому что компаний в отрасли мало",
}

FACTOR_REASON_RU = {
    "last_burst_older_than_18m": "последний судебный всплеск был давно, поэтому его прогностическая сила снижена",
    "last_burst_older_than_12m": "последний судебный всплеск уже не свежий",
    "very_recent_court_activity": "судебная активность совсем свежая",
    "company_is_plaintiff_not_defendant_in_last_burst": "в последнем всплеске компания преимущественно выступала истцом, а не ответчиком",
    "company_mostly_defendant_in_last_burst": "в последнем всплеске компания преимущественно выступала ответчиком",
    "claims_are_tiny_relative_to_assets_and_revenue": "требования к компании как к ответчику очень малы относительно масштаба бизнеса",
    "claims_are_material_relative_to_scale": "требования к компании как к ответчику значимы относительно масштаба бизнеса",
    "claims_are_small_relative_to_tax_paid": "требования к компании как к ответчику малы относительно уплаченных налогов",
    "no_enforcement_and_no_efrsb_context": "нет ни исполнительных производств, ни банкротного контекста ЕФРСБ",
    "has_enforcement_context": "есть исполнительные производства, это усиливает долговой риск",
    "has_efrsb_context": "есть сообщения ЕФРСБ, это усиливает процедурный банкротный риск",
    "recent_case_activity_below_own_history": "последняя судебная активность слабее обычного исторического фона компании",
    "recent_case_activity_above_own_history": "последняя судебная активность выше обычного исторического фона компании",
    "recent_claim_activity_below_own_history": "последние суммы исков ниже обычного исторического фона компании",
    "recent_claim_activity_above_own_history": "последние суммы исков выше обычного исторического фона компании",
}

def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)

def report_slug(inn: str, company_name: str | None) -> str:
    name = safe_filename(company_name or "company").strip("_")[:80] or "company"
    return f"{safe_filename(inn)}_{name}"

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def fetch_url_bytes(url: str, *, timeout: int = 30) -> bytes:
    request = Request(url, headers=HTTP_HEADERS)
    with urlopen(request, timeout=timeout) as response:
        return response.read()

def parse_ru_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("\xa0", "").replace(" ", "").replace(",", ".")
    return bem.safe_float(text)

def is_insurance_company(company_payload: dict[str, Any]) -> bool:
    data = company_payload.get("data") or {}
    okved = data.get("ОКВЭД") or {}
    okved_code = str(okved.get("Код") or "")
    okved_name = str(okved.get("Наим") or "").lower()
    return okved_code.startswith("65.12") or "страхован" in okved_name

def parse_cbr_register_number(html: str) -> str | None:
    match = re.search(
        r"Регистрационный номер</div>\s*<div[^>]*>\s*([^<]+?)\s*</div>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else None

def parse_cbr_capital_ratio(html: str, register_number: str) -> float | None:
    pattern = rf"<td>\s*{re.escape(register_number)}\s*</td>\s*<td>.*?</td>\s*<td>\s*([^<]+?)\s*</td>"
    match = re.search(pattern, html, flags=re.IGNORECASE | re.DOTALL)
    return parse_ru_number(match.group(1)) if match else None

def cbr_xlsx_row_value(zip_file: zipfile.ZipFile, filename: str, register_number: str) -> float | None:
    try:
        import openpyxl
    except ImportError:
        return None

    with zip_file.open(filename) as source:
        workbook = openpyxl.load_workbook(io.BytesIO(source.read()), read_only=True, data_only=True)
    sheet = workbook.active
    try:
        for row in sheet.iter_rows(values_only=True):
            if str(row[0]).strip() == str(register_number):
                return bem.safe_float(row[2])
    finally:
        workbook.close()
    return None

def sum_rub_from_thousand(*values: float | None) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) * 1_000.0

def build_cbr_insurance_context(company_payload: dict[str, Any]) -> dict[str, Any]:
    data = company_payload.get("data") or {}
    ogrn = data.get("ОГРН")
    if not is_insurance_company(company_payload):
        return {"status": "not_insurance"}
    if not ogrn:
        return {"status": "missing_ogrn", "source": "Банк России"}

    card_url = CBR_INSURANCE_CARD_URL.format(ogrn=ogrn)
    try:
        card_html = fetch_url_bytes(card_url).decode("utf-8", errors="ignore")
        register_number = parse_cbr_register_number(card_html)
        if not register_number:
            return {"status": "not_found", "source": "Банк России", "card_url": card_url}

        ratio_html = fetch_url_bytes(CBR_INSURANCE_RATIO_URL).decode("utf-8", errors="ignore")
        capital_ratio = parse_cbr_capital_ratio(ratio_html, register_number)

        archive = fetch_url_bytes(CBR_INSURANCE_PERFORMANCE_ZIP_URL)
        with zipfile.ZipFile(io.BytesIO(archive)) as zip_file:
            life_premiums = cbr_xlsx_row_value(zip_file, "1.xlsx", register_number)
            life_payouts = cbr_xlsx_row_value(zip_file, "2.xlsx", register_number)
            life_net_premiums = cbr_xlsx_row_value(zip_file, "5.xlsx", register_number)
            life_net_payouts = cbr_xlsx_row_value(zip_file, "6.xlsx", register_number)
            non_life_premiums = cbr_xlsx_row_value(zip_file, "7.xlsx", register_number)
            non_life_payouts = cbr_xlsx_row_value(zip_file, "8.xlsx", register_number)
            non_life_net_premiums = cbr_xlsx_row_value(zip_file, "11.xlsx", register_number)
            non_life_net_payouts = cbr_xlsx_row_value(zip_file, "12.xlsx", register_number)

        return {
            "status": "ok",
            "source": "Банк России",
            "report_year": CBR_INSURANCE_REPORT_YEAR,
            "ratio_date": CBR_INSURANCE_RATIO_DATE,
            "card_url": card_url,
            "ratio_url": CBR_INSURANCE_RATIO_URL,
            "performance_url": CBR_INSURANCE_PERFORMANCE_PAGE_URL,
            "cbr_register_number": register_number,
            "capital_obligation_ratio": capital_ratio,
            "gross_premiums_total": sum_rub_from_thousand(life_premiums, non_life_premiums),
            "gross_payouts_total": sum_rub_from_thousand(life_payouts, non_life_payouts),
            "net_premiums_total": sum_rub_from_thousand(life_net_premiums, non_life_net_premiums),
            "net_payouts_total": sum_rub_from_thousand(life_net_payouts, non_life_net_payouts),
            "non_life_gross_premiums": sum_rub_from_thousand(non_life_premiums),
            "non_life_gross_payouts": sum_rub_from_thousand(non_life_payouts),
        }
    except Exception as exc:
        return {
            "status": "error",
            "source": "Банк России",
            "card_url": card_url,
            "error": str(exc),
        }

def is_financial_snapshot_stale(snapshot_year: Any, prediction_date: Any = OBSERVATION_END) -> bool:
    year = bem.safe_float(snapshot_year)
    date = pd.to_datetime(prediction_date, errors="coerce")
    if year is None or pd.isna(date):
        return False
    return int(year) < int(date.year) - MAX_FINANCIAL_SNAPSHOT_AGE_YEARS

def collect_pages(
    endpoint: str,
    inn: str,
    *,
    max_extra_pages: int,
    force: bool,
    company_dir: Path,
    date_from: pd.Timestamp | None = None,
    date_to: pd.Timestamp | None = None,
) -> list[dict[str, Any]]:
    filename = "legal_cases_pages.json" if endpoint == "legal-cases" else "enforcements_pages.json"
    output_file = company_dir / filename
    date_params: dict[str, str] = {}
    if endpoint == "legal-cases":
        if date_from is not None:
            date_params["date_from"] = pd.Timestamp(date_from).date().isoformat()
        if date_to is not None:
            date_params["date_to"] = pd.Timestamp(date_to).date().isoformat()

    if output_file.exists() and not force:
        cached = read_json(output_file)
        cached_meta = cached.get("meta") or {}
        if endpoint != "legal-cases":
            return cached.get("pages") or []
        dates_match = cached_meta.get("date_from") == date_params.get("date_from") and cached_meta.get("date_to") == date_params.get("date_to")
        cached_cap = int(cached_meta.get("max_extra_pages") or -1)
        if dates_match and (not cached_meta.get("truncated_by_max_pages") or cached_cap >= max_extra_pages):
            return cached.get("pages") or []

    pages = [get_json(endpoint, inn=inn, limit=100, page=1, sort="-date", **date_params)]
    total_pages = int(((pages[0].get("data") or {}).get("СтрВсего")) or 1)
    for page in range(2, 2 + min(max(0, total_pages - 1), max_extra_pages)):
        if date_from is not None and pages:
            dates = pd.to_datetime(
                [item.get("Дата") or item.get("ИспПрДата") for item in ((pages[-1].get("data") or {}).get("Записи") or [])],
                errors="coerce",
            )
            if len(dates) and pd.notna(dates.min()) and dates.min() < date_from:
                break
        pages.append(get_json(endpoint, inn=inn, limit=100, page=page, sort="-date", **date_params))
    write_json(
        output_file,
        {
            "pages": pages,
            "meta": {
                "date_from": date_params.get("date_from"),
                "date_to": date_params.get("date_to"),
                "total_pages_reported": total_pages,
                "pages_fetched": len(pages),
                "max_extra_pages": max_extra_pages,
                "truncated_by_max_pages": total_pages > len(pages),
            },
        },
    )
    return pages

def defendant_burst_strength(burst: dict[str, Any]) -> tuple[float, float]:
    return (
        float(burst.get("defendant_claim_sum") or 0.0),
        float(burst.get("defendant_cases") or 0.0),
    )

def collect_company(
    inn: str,
    *,
    force: bool,
    max_extra_case_pages: int,
    max_extra_enforcement_pages: int,
    court_lookback_years: int = DEFAULT_COURT_LOOKBACK_YEARS,
) -> Path:
    company_dir = RAW_DIR / safe_filename(inn)
    company_dir.mkdir(parents=True, exist_ok=True)

    company_file = company_dir / "company.json"
    if force or not company_file.exists():
        write_json(company_file, get_json("company", inn=inn))

    company_payload = read_json(company_file)
    cbr_insurance_file = company_dir / "cbr_insurance_context.json"
    if is_insurance_company(company_payload) and (force or not cbr_insurance_file.exists()):
        write_json(cbr_insurance_file, build_cbr_insurance_context(company_payload))

    finances_file = company_dir / "finances.json"
    if force or not finances_file.exists():
        write_json(finances_file, get_json("finances", inn=inn))

    court_date_from = OBSERVATION_END - pd.DateOffset(years=court_lookback_years)
    court_date_to = OBSERVATION_END
    collect_pages(
        "legal-cases",
        inn,
        max_extra_pages=max_extra_case_pages,
        force=force,
        company_dir=company_dir,
        date_from=court_date_from,
        date_to=court_date_to,
    )
    collect_pages("enforcements", inn, max_extra_pages=max_extra_enforcement_pages, force=force, company_dir=company_dir)
    return company_dir

def company_profile_from_raw(inn: str, company_dir: Path) -> pd.Series:
    payload = read_json(company_dir / "company.json")
    data = payload.get("data") or {}
    status = data.get("Статус") or {}
    liquid = data.get("Ликвид") or {}
    capital = data.get("УстКап") or {}
    taxes = data.get("Налоги") or {}
    return pd.Series(
        {
            "company_inn": inn,
            "status_name": status.get("Наим"),
            "status_code": status.get("Код"),
            "liquid_date": liquid.get("Дата"),
            "liquid_reason": liquid.get("Наим"),
            "region_code_api": (data.get("Регион") or {}).get("Код"),
            "okved_code_api": (data.get("ОКВЭД") or {}).get("Код"),
            "okved_name_api": (data.get("ОКВЭД") or {}).get("Наим"),
            "capital_sum_api": bem.safe_float(capital.get("Сумма")),
            "headcount_api": bem.safe_float(data.get("СЧР")),
            "tax_paid_sum_api": bem.safe_float(taxes.get("СумУпл")),
            "tax_debt_sum_api": bem.safe_float(taxes.get("СумНедоим")),
            "msp_category_api": (data.get("РМСП") or {}).get("Кат"),
            "sanctions_flag_api": data.get("Санкции"),
            "mass_director_flag_api": data.get("МассРуковод"),
            "mass_founder_flag_api": data.get("МассУчред"),
            "disqualified_persons_flag_api": data.get("ДисквЛица"),
        }
    )

def status_gate(company_payload: dict[str, Any]) -> dict[str, Any]:
    data = company_payload.get("data") or {}
    status = data.get("Статус") or {}
    liquid = data.get("Ликвид") or {}
    status_code = str(status.get("Код") or "")
    status_name = str(status.get("Наим") or "")
    liquid_date = liquid.get("Дата")
    liquid_reason = str(liquid.get("Наим") or "")
    text = f"{status_name} {liquid_reason}".lower()

    # Если негативный статус уже известен, это уже не прогноз, а ретроспектива - модель тут не нужна.
    is_active = status_code == "001" and not liquid_date and "действует" in status_name.lower()
    is_negative = bool(liquid_date) or any(keyword in text for keyword in NEGATIVE_STATUS_KEYWORDS)
    if is_negative and not is_active:
        mode = "known_negative"
    elif is_active:
        mode = "predictive"
    else:
        mode = "status_uncertain"

    return {
        "mode": mode,
        "is_active": is_active,
        "is_known_negative": is_negative and not is_active,
        "status_code": status_code or None,
        "status_name": status_name or None,
        "liquid_date": liquid_date,
        "liquid_reason": liquid_reason or None,
    }

def load_company_inputs(
    inn: str,
    company_dir: Path,
    *,
    court_lookback_years: int = DEFAULT_COURT_LOOKBACK_YEARS,
) -> dict[str, Any]:
    original_raw_dir = bem.RAW_RUN_DIR
    try:
        bem.RAW_RUN_DIR = company_dir.parent
        profile = company_profile_from_raw(inn, company_dir)
        raw = read_json(company_dir / "company.json")
        data = raw.get("data") or {}
        reg_date = pd.to_datetime(data.get("ДатаРег"), errors="coerce")
        sample_row = {
            "company_inn": inn,
            "company_name": data.get("НаимСокр") or data.get("НаимПолн"),
            "company_ogrn": data.get("ОГРН"),
            "company_registration_year": float(reg_date.year) if pd.notna(reg_date) else np.nan,
            "region_code": (data.get("Регион") or {}).get("Код"),
            "okved_main": (data.get("ОКВЭД") or {}).get("Код"),
            "authorized_capital": bem.safe_float((data.get("УстКап") or {}).get("Сумма")),
        }

        final_cutoff = OBSERVATION_END
        case_frame = bem.build_case_frame(inn)
        court_date_from = OBSERVATION_END - pd.DateOffset(years=court_lookback_years)
        # Для сервиса берем свежую историю к текущей дате, а не учебный anchor из датасета.
        case_frame = case_frame[case_frame["case_date"] >= court_date_from].copy()
        enforcement_frame = bem.build_enforcement_frame(inn)
        efrsb_frame = bem.build_efrsb_frame(inn)
        finances = bem.parse_financial_years(inn)
        return {
            "profile": profile,
            "raw": raw,
            "data": data,
            "sample_row": sample_row,
            "case_frame": case_frame,
            "enforcement_frame": enforcement_frame,
            "efrsb_frame": efrsb_frame,
            "finances": finances,
        }
    finally:
        bem.RAW_RUN_DIR = original_raw_dir

def build_scoring_row(inn: str, company_dir: Path, *, court_lookback_years: int = DEFAULT_COURT_LOOKBACK_YEARS) -> dict[str, Any]:
    inputs = load_company_inputs(inn, company_dir, court_lookback_years=court_lookback_years)
    try:
        profile = inputs["profile"]
        sample_row = inputs["sample_row"]
        case_frame = inputs["case_frame"]
        enforcement_frame = inputs["enforcement_frame"]
        efrsb_frame = inputs["efrsb_frame"]
        finances = inputs["finances"]
        final_cutoff = OBSERVATION_END
        case_pre_cutoff = case_frame[case_frame["case_date"] <= final_cutoff].copy()
        bursts = bem.build_bursts(case_pre_cutoff)
        last_burst = bursts[-1] if bursts else None
        maxsafe_burst = max(bursts, key=defendant_burst_strength, default=None)
        prediction_date = final_cutoff
        last_burst_anchor_date = pd.Timestamp(last_burst["end_date"]) if last_burst is not None else pd.NaT
        maxsafe_anchor_date = pd.Timestamp(maxsafe_burst["end_date"]) if maxsafe_burst is not None else pd.NaT
        last_defendant_claim = bem.safe_float(last_burst.get("defendant_claim_sum") if last_burst else None) or 0.0
        maxsafe_defendant_claim = bem.safe_float(maxsafe_burst.get("defendant_claim_sum") if maxsafe_burst else None) or 0.0
        original_raw_dir = bem.RAW_RUN_DIR
        bem.RAW_RUN_DIR = company_dir.parent
        try:
            # В сервисе собираю ровно те же признаки, что были при обучении; иначе демо было бы красивым, но нечестным.
            row = {
                **sample_row,
                **bem.prefixed_profile(profile),
                **bem.raw_profile_features(inn),
                **bem.burst_features(last_burst, "last_burst"),
                **bem.burst_features(maxsafe_burst, "maxsafe"),
                **bem.snapshot_features(finances, prediction_date, "last_anchor_fin"),
                **bem.enforcement_features(enforcement_frame, prediction_date, "last_anchor_enf"),
                **bem.efrsb_features(efrsb_frame, prediction_date, "last_anchor_efrsb"),
                **bem.aggregate_case_window(case_frame[case_frame["case_date"] <= prediction_date], prediction_date, 12, "last_anchor_window12"),
                **bem.aggregate_case_window(case_frame[case_frame["case_date"] <= prediction_date], prediction_date, 18, "last_anchor_window18"),
                **bem.aggregate_case_window(case_frame[case_frame["case_date"] <= prediction_date], prediction_date, 24, "last_anchor_window24"),
                **bem.aggregate_previous_case_window(case_frame[case_frame["case_date"] <= prediction_date], prediction_date, 12, "last_anchor_prev12"),
                **bem.aggregate_case_window(
                    case_frame[case_frame["case_date"] <= maxsafe_anchor_date],
                    maxsafe_anchor_date,
                    12,
                    "maxsafe_window12",
                ),
                **bem.aggregate_case_window(
                    case_frame[case_frame["case_date"] <= maxsafe_anchor_date],
                    maxsafe_anchor_date,
                    24,
                    "maxsafe_window24",
                ),
                "service_prediction_date": prediction_date,
                "last_burst_anchor_date": last_burst_anchor_date,
                "maxsafe_anchor_date": maxsafe_anchor_date,
                "safe_history_max_to_last_defendant_claim_ratio": maxsafe_defendant_claim / max(last_defendant_claim, 1.0),
            }
            row |= bem.case_window_comparison_features(row, "last_anchor_window12", "last_anchor_prev12", "last_anchor_window12_vs_prev12")
        finally:
            bem.RAW_RUN_DIR = original_raw_dir
        return row
    except KeyError:
        return {}

def add_scale_normalized_features(row: dict[str, Any]) -> dict[str, Any]:
    claim_sum = bem.safe_float(row.get("last_burst_defendant_claim_sum")) or 0.0
    cases = bem.safe_float(row.get("last_burst_defendant_cases")) or 0.0
    total_claim_sum = bem.safe_float(row.get("last_burst_total_claim_sum")) or 0.0
    total_cases = bem.safe_float(row.get("last_burst_total_cases")) or 0.0
    capital = bem.safe_float(row.get("profile_capital_sum")) or bem.safe_float(row.get("authorized_capital")) or 0.0
    tax_paid = bem.safe_float(row.get("profile_tax_paid_sum")) or 0.0
    stale_finance = is_financial_snapshot_stale(row.get("last_anchor_fin_snapshot_year"), row.get("service_prediction_date"))
    assets = 0.0 if stale_finance else (bem.safe_float(row.get("last_anchor_fin_assets_t0")) or 0.0)
    revenue = 0.0 if stale_finance else (bem.safe_float(row.get("last_anchor_fin_revenue_t0")) or 0.0)
    headcount = bem.safe_float(row.get("profile_headcount")) or 0.0

    return {
        "claim_to_capital": claim_sum / capital if capital else None,
        "claim_to_tax_paid": claim_sum / tax_paid if tax_paid else None,
        "claim_to_assets": claim_sum / assets if assets else None,
        "claim_to_revenue": claim_sum / revenue if revenue else None,
        "cases_to_headcount": cases / headcount if headcount else None,
        "cases_to_tax_paid_mln": cases / (tax_paid / 1_000_000.0) if tax_paid else None,
        "litigation_claim_to_capital": total_claim_sum / capital if capital else None,
        "litigation_cases_to_headcount": total_cases / headcount if headcount else None,
    }

def court_history_context(inn: str, company_dir: Path, *, court_lookback_years: int = DEFAULT_COURT_LOOKBACK_YEARS) -> dict[str, Any]:
    original_raw_dir = bem.RAW_RUN_DIR
    try:
        bem.RAW_RUN_DIR = company_dir.parent
        case_frame = bem.build_case_frame(inn)
        court_date_from = OBSERVATION_END - pd.DateOffset(years=court_lookback_years)
        case_frame = case_frame[case_frame["case_date"] >= court_date_from].copy()
    finally:
        bem.RAW_RUN_DIR = original_raw_dir

    if case_frame.empty:
        return {
            "total_cases_fetched": 0,
            "first_case_date": None,
            "last_case_date": None,
            "active_months": 0,
            "continuous_litigation_baseline": False,
            "windows": {},
        }

    monthly = (
        case_frame.groupby("case_month", as_index=False)
        .agg(cases=("case_uuid", "nunique"), claim_sum=("claim_amount", "sum"))
        .sort_values("case_month")
    )
    anchor = monthly["case_month"].max()
    active_months = int(monthly["case_month"].nunique())
    span_months = bem.month_diff(monthly["case_month"].min(), monthly["case_month"].max()) + 1
    monthly_coverage = active_months / max(span_months, 1)
    windows: dict[str, Any] = {}
    for months in (6, 12, 18, 24, 36):
        recent = monthly[(monthly["case_month"] <= anchor) & (monthly["case_month"] > anchor - pd.DateOffset(months=months))]
        history = monthly[monthly["case_month"] <= anchor - pd.DateOffset(months=months)]
        recent_cases = float(recent["cases"].mean()) if not recent.empty else 0.0
        history_cases = float(history["cases"].mean()) if not history.empty else 0.0
        recent_claim = float(recent["claim_sum"].mean()) if not recent.empty else 0.0
        history_claim = float(history["claim_sum"].mean()) if not history.empty else 0.0
        windows[f"{months}m"] = {
            "recent_cases_per_month": recent_cases,
            "history_cases_per_month": history_cases,
            "case_activity_ratio": recent_cases / history_cases if history_cases else None,
            "recent_claim_per_month": recent_claim,
            "history_claim_per_month": history_claim,
            "claim_activity_ratio": recent_claim / history_claim if history_claim else None,
        }

    return {
        "total_cases_fetched": int(case_frame["case_uuid"].nunique()),
        "first_case_date": str(case_frame["case_date"].min().date()),
        "last_case_date": str(case_frame["case_date"].max().date()),
        "active_months": active_months,
        "span_months": int(span_months),
        "monthly_coverage": monthly_coverage,
        "monthly_cases_p50": float(monthly["cases"].quantile(0.50)),
        "monthly_cases_p90": float(monthly["cases"].quantile(0.90)),
        "monthly_cases_p95": float(monthly["cases"].quantile(0.95)),
        "continuous_litigation_baseline": bool(active_months >= 36 and monthly_coverage >= 0.60),
        "windows": windows,
    }

def industry_baseline(row: dict[str, Any]) -> dict[str, Any]:
    experiment_file = DATA_DIR / "model_dataset.csv"
    if not experiment_file.exists():
        return {}
    frame = pd.read_csv(experiment_file, low_memory=False).copy()
    # Отрасль здесь только справочный фон для отчета. В финальную ML-вероятность ОКВЭД не идет.
    okved = str(row.get("profile_okved_code") or row.get("okved_main") or "")[:2]
    okved_source = frame["profile_okved_code"].where(frame["profile_okved_code"].notna(), frame["okved_main"])
    frame = frame.assign(okved2=okved_source.astype(str).str[:2])
    subset = frame[frame["okved2"] == okved]
    baseline_scope = "industry"
    original_industry_size = int(len(subset))
    if len(subset) < 20:
        subset = frame
        baseline_scope = "global_fallback"

    cases = bem.safe_float(row.get("last_burst_total_cases")) or 0.0
    claim = bem.safe_float(row.get("last_burst_total_claim_sum")) or 0.0
    case_p90 = pd.to_numeric(subset["last_burst_total_cases"], errors="coerce").quantile(0.90)
    claim_p90 = pd.to_numeric(subset["last_burst_total_claim_sum"], errors="coerce").quantile(0.90)
    return {
        "industry_okved2": okved or None,
        "industry_sample_size": original_industry_size,
        "baseline_scope": baseline_scope,
        "baseline_sample_size": int(len(subset)),
        "industry_case_p90": float(case_p90) if pd.notna(case_p90) else None,
        "industry_claim_p90": float(claim_p90) if pd.notna(claim_p90) else None,
        "case_to_industry_p90": cases / case_p90 if case_p90 and pd.notna(case_p90) else None,
        "claim_to_industry_p90": claim / claim_p90 if claim_p90 and pd.notna(claim_p90) else None,
    }

def file_cache_updated_at(path: Path) -> str | None:
    if not path.exists():
        return None
    return str(pd.Timestamp(path.stat().st_mtime, unit="s").floor("s"))

def build_source_freshness(
    company_dir: Path,
    *,
    court_lookback_years: int,
    case_frame: pd.DataFrame,
    enforcement_frame: pd.DataFrame,
    efrsb_frame: pd.DataFrame,
    finances: dict[int, dict[str, float]],
) -> dict[str, Any]:
    company_file = company_dir / "company.json"
    cases_file = company_dir / "legal_cases_pages.json"
    finances_file = company_dir / "finances.json"
    cbr_insurance_file = company_dir / "cbr_insurance_context.json"
    enforcements_file = company_dir / "enforcements_pages.json"
    cases_meta = (read_json(cases_file).get("meta") or {}) if cases_file.exists() else {}
    cbr_insurance = read_json(cbr_insurance_file)

    sources = {
        "company_profile": {
            "loaded": company_file.exists(),
            "cache_updated_at": file_cache_updated_at(company_file),
            "latest_record_date": None,
            "records_found": None,
            "note": "Карточка компании и текущий статус",
        },
        "legal_cases": {
            "loaded": cases_file.exists(),
            "cache_updated_at": file_cache_updated_at(cases_file),
            "latest_record_date": str(case_frame["case_date"].max().date()) if not case_frame.empty else None,
            "records_found": int(case_frame["case_uuid"].nunique()) if not case_frame.empty else 0,
            "lookback_years": court_lookback_years,
            "date_from": cases_meta.get("date_from"),
            "date_to": cases_meta.get("date_to"),
            "pages_fetched": cases_meta.get("pages_fetched"),
            "total_pages_reported": cases_meta.get("total_pages_reported"),
            "truncated_by_max_pages": cases_meta.get("truncated_by_max_pages"),
            "note": "Арбитражные дела в окне наблюдения",
        },
        "finances": {
            "loaded": finances_file.exists(),
            "cache_updated_at": file_cache_updated_at(finances_file),
            "latest_report_year": max(finances) if finances else None,
            "records_found": int(len(finances)),
            "note": "Опциональный финансовый контекст; не используется в ML-score",
        },
        "enforcements": {
            "loaded": enforcements_file.exists(),
            "cache_updated_at": file_cache_updated_at(enforcements_file),
            "latest_record_date": str(enforcement_frame["enforcement_date"].max().date()) if not enforcement_frame.empty else None,
            "records_found": int(enforcement_frame["enforcement_number"].nunique()) if not enforcement_frame.empty else 0,
            "note": "Исполнительные производства",
        },
        "efrsb": {
            "loaded": company_file.exists(),
            "cache_updated_at": file_cache_updated_at(company_file),
            "latest_record_date": str(efrsb_frame["message_date"].max().date()) if not efrsb_frame.empty else None,
            "records_found": int(len(efrsb_frame)),
            "note": "Сообщения ЕФРСБ из карточки компании",
        },
    }
    if cbr_insurance_file.exists():
        sources["cbr_insurance"] = {
            "loaded": True,
            "cache_updated_at": file_cache_updated_at(cbr_insurance_file),
            "latest_report_year": cbr_insurance.get("report_year"),
            "records_found": 1 if cbr_insurance.get("status") == "ok" else 0,
            "note": "Страховой контекст Банка России; не используется в ML-score",
        }
    return sources

def build_data_quality(
    company_dir: Path,
    row: dict[str, Any],
    burst: dict[str, Any],
    court_history: dict[str, Any],
    industry: dict[str, Any],
) -> dict[str, Any]:
    cases_meta = read_json(company_dir / "legal_cases_pages.json").get("meta") or {}
    cases_truncated = bool(cases_meta.get("truncated_by_max_pages"))
    financial_snapshot_present = bool(int(row.get("last_anchor_fin_has_snapshot") or 0))
    financial_snapshot_stale = financial_snapshot_present and is_financial_snapshot_stale(
        row.get("last_anchor_fin_snapshot_year"),
        row.get("service_prediction_date"),
    )
    source_blocks = {
        "company_profile": (company_dir / "company.json").exists(),
        "legal_cases": (company_dir / "legal_cases_pages.json").exists(),
        "finances": (company_dir / "finances.json").exists(),
        "enforcements": (company_dir / "enforcements_pages.json").exists(),
        "efrsb": (company_dir / "company.json").exists(),
    }
    analytic_blocks = {
        "company_profile": any(
            pd.notna(row.get(field))
            for field in ("company_registration_year", "profile_okved_code", "region_code")
        ),
        "court_history": has_court_history(court_history),
        "court_history_complete": has_court_history(court_history) and not cases_truncated,
        "last_burst": has_burst(burst),
        "industry_baseline": bool(industry.get("baseline_sample_size")),
        "procedural_context_source": source_blocks["enforcements"] and source_blocks["efrsb"],
    }
    context_blocks = {
        "financial_context": financial_snapshot_present,
        "financial_context_current": financial_snapshot_present and not financial_snapshot_stale,
    }

    source_score = sum(source_blocks.values()) / max(len(source_blocks), 1)
    analytic_score = sum(analytic_blocks.values()) / max(len(analytic_blocks), 1)
    overall_score = 0.4 * source_score + 0.6 * analytic_score
    if cases_truncated:
        overall_score = min(overall_score, 0.60)

    if overall_score >= 0.85:
        confidence_level = "high"
        confidence_label = "высокая"
    elif overall_score >= 0.65:
        confidence_level = "medium"
        confidence_label = "средняя"
    else:
        confidence_level = "low"
        confidence_label = "низкая"

    missing_analytic = [name for name, flag in analytic_blocks.items() if not flag]
    notes: list[str] = []
    context_notes: list[str] = []
    cbr_insurance = read_json(company_dir / "cbr_insurance_context.json")
    if not analytic_blocks["last_burst"]:
        notes.append("последний судебный всплеск не сформирован")
    if not context_blocks["financial_context"]:
        if cbr_insurance.get("status") == "ok":
            context_notes.append("стандартная финансовая отчетность не найдена; для страховщика добавлен контекст Банка России")
        else:
            context_notes.append("стандартная финансовая отчетность не найдена; используется профильный масштаб компании")
    if financial_snapshot_stale:
        context_notes.append(f"финансовый снимок устарел: {fmt_value(row.get('last_anchor_fin_snapshot_year'))}")
    if not analytic_blocks["court_history"]:
        notes.append("судебный исторический фон ограничен или отсутствует")
    if cases_truncated:
        notes.append(
            f"судебная выдача обрезана: загружено {fmt_value(cases_meta.get('pages_fetched'))} страниц из {fmt_value(cases_meta.get('total_pages_reported'))}"
        )

    return {
        "source_coverage_score": round(source_score, 4),
        "analytic_coverage_score": round(analytic_score, 4),
        "overall_score": round(overall_score, 4),
        "source_coverage_percent": round(source_score * 100, 1),
        "analytic_coverage_percent": round(analytic_score * 100, 1),
        "overall_percent": round(overall_score * 100, 1),
        "confidence_level": confidence_level,
        "confidence_label": confidence_label,
        "source_blocks": source_blocks,
        "analytic_blocks": analytic_blocks,
        "context_blocks": context_blocks,
        "missing_analytic_blocks": missing_analytic,
        "notes": notes,
        "context_notes": context_notes,
        "blocking_limitations": {
            "legal_cases_truncated": cases_truncated,
        },
    }

def build_result_state(
    gate: dict[str, Any],
    score: dict[str, Any],
    data_quality: dict[str, Any],
    burst: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    if gate["mode"] == "known_negative":
        return {
            "code": "known_negative",
            "label": "известен негативный статус",
            "message": "Компания уже находится в негативном или ликвидационном статусе, поэтому прогнозная оценка не является основным результатом.",
        }
    if gate["mode"] == "status_uncertain":
        return {
            "code": "status_uncertain",
            "label": "неопределенный статус",
            "message": "Статус компании недостаточно определен для уверенной прогнозной трактовки.",
        }
    if score.get("probability") is None or data_quality["overall_score"] < 0.45 or (data_quality.get("blocking_limitations") or {}).get("legal_cases_truncated"):
        # Лучше честно сказать "данных мало", чем красиво нарисовать вероятность на обрезанной судебной выдаче.
        return {
            "code": "insufficient_data",
            "label": "недостаточно данных",
            "message": "Источников или аналитических блоков недостаточно для уверенной интерпретации результата.",
        }
    no_burst = not has_burst(burst)
    no_proc_context = (bem.safe_float(context.get("enforcement_count")) or 0.0) == 0 and (bem.safe_float(context.get("efrsb_count")) or 0.0) == 0
    if no_burst and no_proc_context:
        return {
            "code": "no_signal",
            "label": "выраженный стресс-сигнал не найден",
            "message": "По доступным данным не выявлен выраженный судебный всплеск или поддерживающий процедурный контекст, поэтому результат следует читать как отсутствие сильного наблюдаемого сигнала, а не как гарантию устойчивости.",
        }
    if score["probability"] >= score["threshold"] and score.get("risk_level") == "high":
        return {
            "code": "risk_detected",
            "label": "высокий риск выявлен",
            "message": "Калиброванная вероятность заметно выше рабочего порога модели; судебная динамика требует отдельной проверки.",
        }
    if score["probability"] >= score["threshold"]:
        return {
            "code": "risk_watch",
            "label": "пограничный риск",
            "message": "Калиброванная вероятность выше рабочего порога, но ниже зоны высокого риска; результат следует читать как сигнал к ручной проверке.",
        }
    return {
        "code": "no_material_risk",
        "label": "выраженный риск не выявлен",
        "message": "По текущим данным риск ниже рабочего порога модели, однако его следует трактовать вместе с качеством покрытия данных и характером сигнала.",
    }

def build_timeline(
    case_frame: pd.DataFrame,
    enforcement_frame: pd.DataFrame,
    efrsb_frame: pd.DataFrame,
    burst: dict[str, Any],
    *,
    months_back: int = 24,
) -> dict[str, Any]:
    candidates: list[pd.Timestamp] = []
    if not case_frame.empty:
        candidates.append(pd.Timestamp(case_frame["case_date"].max()))
    if not enforcement_frame.empty:
        candidates.append(pd.Timestamp(enforcement_frame["enforcement_date"].max()))
    if not efrsb_frame.empty:
        candidates.append(pd.Timestamp(efrsb_frame["message_date"].max()))
    if has_burst(burst):
        candidates.append(pd.to_datetime(burst.get("end_date"), errors="coerce"))

    focus_end = max([item for item in candidates if pd.notna(item)], default=OBSERVATION_END)
    focus_month = pd.Timestamp(focus_end).to_period("M").to_timestamp()
    months = pd.date_range(end=focus_month, periods=months_back, freq="MS")

    case_monthly = pd.DataFrame(columns=["month", "cases", "claim_sum", "defendant_cases", "plaintiff_cases"])
    if not case_frame.empty:
        case_monthly = (
            case_frame.groupby("case_month", as_index=False)
            .agg(
                cases=("case_uuid", "nunique"),
                claim_sum=("claim_amount", "sum"),
                defendant_cases=("role", lambda s: int((s == "defendant").sum())),
                plaintiff_cases=("role", lambda s: int((s == "plaintiff").sum())),
            )
            .rename(columns={"case_month": "month"})
        )

    enforcement_monthly = pd.DataFrame(columns=["month", "enforcement_count", "enforcement_debt"])
    if not enforcement_frame.empty:
        temp = enforcement_frame.copy()
        temp["month"] = temp["enforcement_date"].dt.to_period("M").dt.to_timestamp()
        enforcement_monthly = temp.groupby("month", as_index=False).agg(
            enforcement_count=("enforcement_number", "nunique"),
            enforcement_debt=("debt_sum", "sum"),
        )

    efrsb_monthly = pd.DataFrame(columns=["month", "efrsb_count"])
    if not efrsb_frame.empty:
        temp = efrsb_frame.copy()
        temp["month"] = temp["message_date"].dt.to_period("M").dt.to_timestamp()
        efrsb_monthly = temp.groupby("month", as_index=False).agg(efrsb_count=("message_type", "count"))

    burst_start = pd.to_datetime(burst.get("start_date"), errors="coerce")
    burst_end = pd.to_datetime(burst.get("end_date"), errors="coerce")

    rows: list[dict[str, Any]] = []
    for month in months:
        case_row = case_monthly.loc[case_monthly["month"] == month]
        enf_row = enforcement_monthly.loc[enforcement_monthly["month"] == month]
        efrsb_row = efrsb_monthly.loc[efrsb_monthly["month"] == month]
        row = {
            "month": str(month.date()),
            "cases": int(case_row["cases"].iloc[0]) if not case_row.empty else 0,
            "claim_sum": float(case_row["claim_sum"].iloc[0]) if not case_row.empty else 0.0,
            "defendant_cases": int(case_row["defendant_cases"].iloc[0]) if not case_row.empty else 0,
            "plaintiff_cases": int(case_row["plaintiff_cases"].iloc[0]) if not case_row.empty else 0,
            "enforcement_count": int(enf_row["enforcement_count"].iloc[0]) if not enf_row.empty else 0,
            "enforcement_debt": float(enf_row["enforcement_debt"].iloc[0]) if not enf_row.empty else 0.0,
            "efrsb_count": int(efrsb_row["efrsb_count"].iloc[0]) if not efrsb_row.empty else 0,
            "in_burst": bool(pd.notna(burst_start) and pd.notna(burst_end) and month >= burst_start.to_period("M").to_timestamp() and month <= burst_end.to_period("M").to_timestamp()),
        }
        row["has_signal"] = bool(
            row["cases"] > 0
            or row["enforcement_count"] > 0
            or row["efrsb_count"] > 0
            or row["in_burst"]
        )
        rows.append(row)

    return {
        "focus_end": str(pd.Timestamp(focus_end).date()),
        "months_back": months_back,
        "rows": rows,
        "nonzero_rows": [row for row in rows if row["has_signal"]],
    }

def build_executive_summary(
    gate: dict[str, Any],
    result_state: dict[str, Any],
    score: dict[str, Any],
    data_quality: dict[str, Any],
    burst: dict[str, Any],
    context: dict[str, Any],
    industry: dict[str, Any],
    scale: dict[str, Any],
) -> list[str]:
    summary = [result_state["message"]]

    if has_burst(burst):
        summary.append(
            f"Последний судебный всплеск длился {fmt_value(burst.get('months'))} {plural(burst.get('months'), ('месяц', 'месяца', 'месяцев'))}, включал {fmt_value(burst.get('total_cases'))} {plural(burst.get('total_cases'), ('дело', 'дела', 'дел'))} и {fmt_money(burst.get('total_claim_sum'))} суммарных требований."
        )
    else:
        summary.append("Судебный всплеск в выбранном окне наблюдения не сформирован.")

    if (bem.safe_float(context.get("enforcement_count")) or 0.0) > 0:
        summary.append(
            f"Есть исполнительные производства ({fmt_value(context.get('enforcement_count'))}) на сумму {fmt_money(context.get('enforcement_debt'))}."
        )
    elif (bem.safe_float(context.get("efrsb_count")) or 0.0) > 0:
        summary.append(
            f"Исполнительных производств не найдено, но есть процедурный банкротный контекст ЕФРСБ ({fmt_value(context.get('efrsb_count'))} сообщений)."
        )
    else:
        summary.append("Поддерживающий долговой или банкротный контекст по ФССП/ЕФРСБ не выражен.")

    if (industry.get("case_to_industry_p90") or 0) > 1:
        summary.append("Нагрузка по числу дел превышает высокий уровень для отраслевой группы.")
    elif (industry.get("claim_to_industry_p90") or 0) > 1:
        summary.append("Нагрузка по сумме исков превышает высокий уровень для отраслевой группы.")
    else:
        summary.append("По сравнению с похожими компаниями нагрузка не выглядит экстремальной.")

    if data_quality["confidence_level"] != "high":
        summary.append(
            f"Уверенность результата {data_quality['confidence_label']}: покрытие данных {data_quality['overall_percent']}%, ограничения: {'; '.join(data_quality.get('notes') or []) or 'нет'}."
        )
    else:
        summary.append(f"Уверенность результата высокая: покрытие данных составляет {data_quality['overall_percent']}%.")

    if score.get("probability") is not None:
        summary.append(
            f"Итоговая вероятность риска составляет {fmt_probability(score.get('probability'))} при рабочем пороге {fmt_probability(score.get('threshold'))}."
        )

    if (scale.get("claim_to_assets") or 0) > 0.25 or (scale.get("claim_to_revenue") or 0) > 0.15:
        summary.append("Требования к компании как к ответчику материальны относительно масштаба бизнеса, что усиливает интерпретацию стресса.")

    return summary[:6]

def load_service_columns() -> list[str]:
    if SERVICE_FEATURES_FILE.exists():
        return json.loads(SERVICE_FEATURES_FILE.read_text(encoding="utf-8"))["features"]
    raise FileNotFoundError(
        f"Service feature list is missing: {SERVICE_FEATURES_FILE}. "
        "Expected a shipped final feature allowlist in recovery_index/models/."
    )

def score_predictive(row: dict[str, Any], service_columns: list[str]) -> float:
    model = joblib.load(MODEL_FILE)
    frame = pd.DataFrame([row])
    missing = {column: np.nan for column in service_columns if column not in frame.columns}
    if missing:
        frame = pd.concat([frame, pd.DataFrame([missing])], axis=1)
    # Порядок колонок беру из сохраненного allowlist - это мелочь, но без нее sklearn легко уедет.
    return float(model.predict_proba(frame[service_columns])[:, 1][0])

def risk_level(probability: float | None, threshold: float) -> str:
    if probability is None:
        return "not_scored"
    # Порог модели и "высокая зона" разведены: пограничный сигнал не должен звучать как финальный приговор.
    if probability >= max(0.55, threshold + 0.12):
        return "high"
    if probability >= threshold:
        return "watch"
    if probability >= max(0.35, threshold - 0.05):
        return "medium"
    return "low"

def risk_level_ru(code: str) -> str:
    return RISK_LEVEL_RU.get(code, code)

def gate_mode_ru(code: str) -> str:
    return GATE_MODE_RU.get(code, code)

def target_ru(code: str) -> str:
    return TARGET_RU.get(code, code)

def baseline_scope_ru(code: str) -> str:
    return BASELINE_SCOPE_RU.get(code, code)

def yes_no(value: Any) -> str:
    if value is None:
        return "нет данных"
    return "да" if bool(value) else "нет"

def fmt_probability(value: Any) -> str:
    number = bem.safe_float(value)
    if number is None:
        return "нет данных"
    return f"{number:.1%}"

def plural(value: Any, forms: tuple[str, str, str]) -> str:
    number = bem.safe_float(value)
    if number is None:
        return forms[2]
    n = abs(int(number))
    if 11 <= n % 100 <= 14:
        return forms[2]
    if n % 10 == 1:
        return forms[0]
    if n % 10 in {2, 3, 4}:
        return forms[1]
    return forms[2]

def fmt_pp(value: Any) -> str:
    number = bem.safe_float(value)
    if number is None:
        return "нет данных"
    pp = number * 100
    sign = "+" if pp > 0 else ""
    rendered = f"{pp:.0f}" if float(pp).is_integer() else f"{pp:.1f}".rstrip("0").rstrip(".")
    return f"{sign}{rendered} п.п."

def explain_factor(factor: dict[str, Any]) -> str:
    direction = factor.get("direction")
    direction_ru = "указывает на пониженный риск" if direction == "down" else "указывает на повышенный риск"
    reason_ru = FACTOR_REASON_RU.get(factor.get("reason"), str(factor.get("reason") or "фактор модели"))
    return f"- {direction_ru}: {reason_ru}."

def months_since(date_value: Any, current_date: pd.Timestamp = OBSERVATION_END) -> float | None:
    date = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(date):
        return None
    return max(0.0, (current_date - date).days / 30.44)

def build_signal_context(
    probability: float | None,
    row: dict[str, Any],
    scale: dict[str, Any],
    court_history: dict[str, Any],
) -> dict[str, Any]:
    if probability is None:
        return {"factors": []}

    factors: list[dict[str, Any]] = []

    burst_end = row.get("last_burst_end_date")
    age_months = months_since(burst_end)
    defendant_cases = bem.safe_float(row.get("last_burst_defendant_cases")) or 0.0
    plaintiff_cases = bem.safe_float(row.get("last_burst_plaintiff_cases")) or 0.0
    total_cases = bem.safe_float(row.get("last_burst_total_cases")) or 0.0
    defendant_share = defendant_cases / total_cases if total_cases else 0.0
    plaintiff_share = plaintiff_cases / total_cases if total_cases else 0.0
    claim_to_assets = scale.get("claim_to_assets")
    claim_to_revenue = scale.get("claim_to_revenue")
    claim_to_tax_paid = scale.get("claim_to_tax_paid")
    enf_count = bem.safe_float(row.get("last_anchor_enf_count_total")) or 0.0
    efrsb_count = bem.safe_float(row.get("last_anchor_efrsb_total_count")) or 0.0
    ratio_12m = ((court_history.get("windows") or {}).get("12m") or {}).get("case_activity_ratio")
    claim_ratio_12m = ((court_history.get("windows") or {}).get("12m") or {}).get("claim_activity_ratio")

    if age_months is not None:
        if age_months > 18:
            factors.append({"direction": "down", "reason": "last_burst_older_than_18m", "value": round(age_months, 2)})
        elif age_months > 12:
            factors.append({"direction": "down", "reason": "last_burst_older_than_12m", "value": round(age_months, 2)})
        elif age_months <= 3:
            factors.append({"direction": "up", "reason": "very_recent_court_activity", "value": round(age_months, 2)})

    if total_cases and defendant_share == 0 and plaintiff_share >= 0.8:
        factors.append({"direction": "down", "reason": "company_is_plaintiff_not_defendant_in_last_burst", "value": plaintiff_share})
    elif defendant_share >= 0.7:
        factors.append({"direction": "up", "reason": "company_mostly_defendant_in_last_burst", "value": defendant_share})

    if (claim_to_assets is not None and claim_to_assets < 0.01) and (claim_to_revenue is not None and claim_to_revenue < 0.01):
        factors.append({"direction": "down", "reason": "claims_are_tiny_relative_to_assets_and_revenue", "value": {"claim_to_assets": claim_to_assets, "claim_to_revenue": claim_to_revenue}})
    elif (claim_to_assets is not None and claim_to_assets > 0.25) or (claim_to_revenue is not None and claim_to_revenue > 0.15):
        factors.append({"direction": "up", "reason": "claims_are_material_relative_to_scale", "value": {"claim_to_assets": claim_to_assets, "claim_to_revenue": claim_to_revenue}})

    if claim_to_tax_paid is not None and claim_to_tax_paid < 0.05:
        factors.append({"direction": "down", "reason": "claims_are_small_relative_to_tax_paid", "value": claim_to_tax_paid})

    if enf_count == 0 and efrsb_count == 0:
        factors.append({"direction": "down", "reason": "no_enforcement_and_no_efrsb_context", "value": {"enforcement": enf_count, "efrsb": efrsb_count}})
    else:
        if enf_count > 0:
            factors.append({"direction": "up", "reason": "has_enforcement_context", "value": enf_count})
        if efrsb_count > 0:
            factors.append({"direction": "up", "reason": "has_efrsb_context", "value": efrsb_count})

    if ratio_12m is not None and ratio_12m < 1.0:
        factors.append({"direction": "down", "reason": "recent_case_activity_below_own_history", "value": ratio_12m})
    elif ratio_12m is not None and ratio_12m > 2.0:
        factors.append({"direction": "up", "reason": "recent_case_activity_above_own_history", "value": ratio_12m})

    if claim_ratio_12m is not None and claim_ratio_12m < 0.5:
        factors.append({"direction": "down", "reason": "recent_claim_activity_below_own_history", "value": claim_ratio_12m})
    elif claim_ratio_12m is not None and claim_ratio_12m > 2.0:
        factors.append({"direction": "up", "reason": "recent_claim_activity_above_own_history", "value": claim_ratio_12m})

    return {
        "factors": factors,
        "current_date": str(OBSERVATION_END.date()),
        "last_burst_age_months": age_months,
        "defendant_share": defendant_share,
        "plaintiff_share": plaintiff_share,
    }

def fmt_money(value: Any) -> str:
    number = bem.safe_float(value)
    if number is None:
        return "нет данных"
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f} млрд руб."
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.2f} млн руб."
    return f"{number:,.0f} руб.".replace(",", " ")

def fmt_date(value: Any) -> str:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return "нет данных"
    return str(date.date())

def fmt_percent(value: Any, digits: int = 1) -> str:
    number = bem.safe_float(value)
    if number is None:
        return "нет данных"
    rendered = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    return f"{rendered}%"

def fmt_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "нет данных"
    if isinstance(value, str):
        parsed_date = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed_date):
            return str(parsed_date.date())
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        if abs(number) >= 10:
            return f"{number:.1f}".rstrip("0").rstrip(".")
        return f"{number:.3f}".rstrip("0").rstrip(".")
    return str(value)

def has_burst(burst: dict[str, Any]) -> bool:
    total_cases = bem.safe_float(burst.get("total_cases")) or 0.0
    start_date = pd.to_datetime(burst.get("start_date"), errors="coerce")
    end_date = pd.to_datetime(burst.get("end_date"), errors="coerce")
    return total_cases > 0 and pd.notna(start_date) and pd.notna(end_date)

def has_court_history(court_history: dict[str, Any]) -> bool:
    return int(court_history.get("total_cases_fetched") or 0) > 0

def court_window_summary(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "count",
        "claim_sum",
        "claim_max",
        "defendant_count",
        "plaintiff_count",
        "defendant_claim_sum",
        "plaintiff_claim_sum",
        "active_months",
    )

    def window(prefix: str) -> dict[str, Any]:
        return {field: bem.safe_float(row.get(f"{prefix}_{field}")) or 0.0 for field in fields}

    # Этот блок нужен не модели, а человеку: быстро видно last12, prev12, last24 и изменение год к году.
    return {
        "last12": window("last_anchor_window12"),
        "previous12": window("last_anchor_prev12"),
        "last24": window("last_anchor_window24"),
        "change": {
            field: {
                "diff": bem.safe_float(row.get(f"last_anchor_window12_vs_prev12_{field}_diff")) or 0.0,
                "log_change": bem.safe_float(row.get(f"last_anchor_window12_vs_prev12_{field}_log_change")) or 0.0,
            }
            for field in fields
        },
    }

def short_summary(assessment: dict[str, Any], report_path: Path) -> str:
    company = assessment["company"]
    score = assessment["score"]
    gate = assessment["status_gate"]
    result_state = assessment.get("result_state") or {}
    data_quality = assessment.get("data_quality") or {}
    context = assessment["context"]
    burst = assessment["last_burst"]
    lines = [
        "",
        "Готово.",
        f"Компания: {company.get('name') or 'нет названия'}",
        f"ИНН: {company.get('inn')}",
        f"Статус: {gate.get('status_name') or 'нет данных'}",
        f"Режим: {gate_mode_ru(gate.get('mode'))}",
        f"Состояние результата: {result_state.get('label') or 'нет данных'}",
        f"Уверенность: {data_quality.get('confidence_label') or 'нет данных'} ({fmt_percent(data_quality.get('overall_percent'))})",
    ]
    if data_quality.get("notes"):
        lines.append(f"Ограничение: {data_quality['notes'][0]}")
    elif data_quality.get("context_notes"):
        lines.append(f"Контекст: {data_quality['context_notes'][0]}")
    if score.get("probability") is None:
        lines.append(result_state.get("message") or "Прогноз не рассчитывается: у компании уже негативный или неопределенный статус.")
    else:
        lines.extend(
            [
                f"Риск негативного исхода в ближайшие 12 месяцев: {fmt_probability(score.get('probability'))} ({risk_level_ru(score['risk_level'])})",
                f"Модель: {score.get('model') or 'risk_model_12m'}; порог риска: {fmt_probability(score.get('threshold'))}",
            ]
        )
    if has_burst(burst):
        burst_line = f"Последний судебный эпизод: {fmt_date(burst.get('end_date'))} | дел: {fmt_value(burst.get('total_cases'))} | сумма: {fmt_money(burst.get('total_claim_sum'))}"
    else:
        burst_line = "Последний судебный эпизод: за выбранный период судебных дел не найдено"
    lines.extend(
        [
            burst_line,
            f"ФССП: {fmt_value(context.get('enforcement_count'))}; ЕФРСБ: {fmt_value(context.get('efrsb_count'))}",
            f"Файл отчета: {report_path}",
            "",
        ]
    )
    return "\n".join(lines)

def build_report(assessment: dict[str, Any]) -> str:
    company = assessment["company"]
    gate = assessment["status_gate"]
    score = assessment["score"]
    result_state = assessment.get("result_state") or {}
    data_quality = assessment.get("data_quality") or {}
    source_freshness = assessment.get("source_freshness") or {}
    executive_summary = assessment.get("executive_summary") or []
    burst = assessment["last_burst"]
    scale = assessment["scale_normalization"]
    industry = assessment["industry_baseline"]
    context = assessment["context"]
    court_history = assessment.get("court_history") or {}
    court_windows = assessment.get("court_windows") or {}
    timeline = assessment.get("timeline") or {"nonzero_rows": []}
    signal_context = assessment.get("signal_context") or {}

    lines = [
        f"# Отчет по компании: {company['name'] or company['inn']}",
        "",
        "## Краткий вывод",
        f"- ИНН: `{company['inn']}`",
        f"- ОГРН: `{company.get('ogrn') or 'нет данных'}`",
        f"- Статус по Ofdata: `{gate.get('status_name') or 'нет данных'}`",
        f"- Режим отчета: `{gate_mode_ru(gate['mode'])}`",
        f"- Состояние результата: `{result_state.get('label') or 'нет данных'}`",
        f"- Качество покрытия данных: `{fmt_percent(data_quality.get('overall_percent'))}` (`{data_quality.get('confidence_label') or 'нет данных'}` уверенность)",
    ]
    if gate["mode"] == "known_negative":
        lines.extend(
            [
                f"- Вывод: {result_state.get('message') or 'компания уже находится в негативном статусе.'}",
                f"- Дата ликвидации: `{gate.get('liquid_date') or 'нет данных'}`",
                f"- Причина: `{gate.get('liquid_reason') or 'нет данных'}`",
            ]
        )
    else:
        lines.extend(
            [
                f"- Дата оценки: `{fmt_date(score.get('prediction_date'))}`",
                f"- Вероятность негативного исхода в ближайшие 12 месяцев: `{fmt_probability(score.get('probability'))}`",
                f"- Модель: `{score.get('model') or 'risk_model_12m'}`",
                f"- Порог для срабатывания риска: `{fmt_probability(score.get('threshold'))}`",
                f"- Итоговый уровень риска: `{risk_level_ru(score.get('risk_level'))}`",
                f"- Интерпретация состояния: `{result_state.get('message') or 'нет данных'}`",
                f"- Что именно прогнозируется: `{target_ru(score.get('target'))}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Ключевые выводы",
            *([f"- {item}" for item in executive_summary] or ["- Краткая интерпретация не сформирована."]),
            "",
            "## Качество покрытия данных",
            f"- Покрытие источников: `{fmt_percent(data_quality.get('source_coverage_percent'))}`",
            f"- Покрытие аналитических блоков: `{fmt_percent(data_quality.get('analytic_coverage_percent'))}`",
            f"- Общая уверенность: `{data_quality.get('confidence_label') or 'нет данных'}`",
            f"- Ограниченные блоки: `{', '.join(data_quality.get('missing_analytic_blocks') or []) or 'нет'}`",
            *([f"- Ограничение: {note}" for note in (data_quality.get('notes') or [])] or ["- Существенных ограничений покрытия не зафиксировано."]),
            *([f"- Отчетный контекст: {note}" for note in (data_quality.get('context_notes') or [])] or []),
        ]
    )

    lines.extend(["", "## Последний судебный всплеск"])
    if has_burst(burst):
        lines.extend(
            [
                f"- Период: `{fmt_date(burst.get('start_date'))}` - `{fmt_date(burst.get('end_date'))}`",
                f"- Длительность: `{fmt_value(burst.get('months'))} {plural(burst.get('months'), ('месяц', 'месяца', 'месяцев'))}`",
                f"- Судебных дел: `{fmt_value(burst.get('total_cases'))} {plural(burst.get('total_cases'), ('дело', 'дела', 'дел'))}`",
                f"- Общая сумма судебных требований: {fmt_money(burst.get('total_claim_sum'))}",
                f"- Дел в месяц: `{fmt_value(burst.get('cases_per_month'))}`",
                f"- Общая сумма судебных требований в месяц: {fmt_money(burst.get('claim_per_month'))}",
                f"- Компания выступала ответчиком: `{fmt_value(burst.get('defendant_cases'))} {plural(burst.get('defendant_cases'), ('раз', 'раза', 'раз'))}`",
                f"- Компания выступала истцом: `{fmt_value(burst.get('plaintiff_cases'))} {plural(burst.get('plaintiff_cases'), ('раз', 'раза', 'раз'))}`",
            ]
        )
    else:
        lines.extend(
            [
                "- Судебных дел за выбранный период не найдено, поэтому судебный всплеск не сформирован.",
                "- Судебные признаки для модели в этой главе считаются отсутствующими/нулевыми, а не неизвестной ошибкой парсинга.",
            ]
        )

    lines.extend(["", "## Исторический судебный фон"])
    if has_court_history(court_history):
        last12 = court_windows.get("last12") or {}
        previous12 = court_windows.get("previous12") or {}
        last24 = court_windows.get("last24") or {}
        change = court_windows.get("change") or {}
        lines.extend(
            [
                f"- Найдено судебных дел: `{fmt_value(court_history.get('total_cases_fetched'))}`",
                *(
                    [
                        f"- Ограничение судебных данных: загружено `{fmt_value((source_freshness.get('legal_cases') or {}).get('pages_fetched'))}` страниц из `{fmt_value((source_freshness.get('legal_cases') or {}).get('total_pages_reported'))}`; история неполная."
                    ]
                    if (source_freshness.get("legal_cases") or {}).get("truncated_by_max_pages")
                    else []
                ),
                f"- Первая и последняя найденные даты: `{fmt_date(court_history.get('first_case_date'))}` - `{fmt_date(court_history.get('last_case_date'))}`",
                f"- Активные календарные месяцы / календарный охват: `{fmt_value(court_history.get('active_months'))}` / `{fmt_value(court_history.get('span_months'))}`",
                f"- Есть ли устойчивый судебный фон: `{yes_no(court_history.get('continuous_litigation_baseline'))}`",
                f"- Ориентиры судебной активности по месяцам: `{fmt_value(court_history.get('monthly_cases_p50'))}` / `{fmt_value(court_history.get('monthly_cases_p90'))}` / `{fmt_value(court_history.get('monthly_cases_p95'))}`",
                f"- Активность дел за последние 12 месяцев относительно собственной истории: `{fmt_value((court_history.get('windows') or {}).get('12m', {}).get('case_activity_ratio'))}`",
                f"- Активность сумм исков за последние 12 месяцев относительно собственной истории: `{fmt_value((court_history.get('windows') or {}).get('12m', {}).get('claim_activity_ratio'))}`",
                f"- Последние 12 месяцев: `{fmt_value(last12.get('count'))}` дел, `{fmt_money(last12.get('claim_sum'))}` требований.",
                f"- Предыдущие 12 месяцев: `{fmt_value(previous12.get('count'))}` дел, `{fmt_money(previous12.get('claim_sum'))}` требований.",
                f"- Все 24 месяца: `{fmt_value(last24.get('count'))}` дел, `{fmt_money(last24.get('claim_sum'))}` требований.",
                f"- Изменение последнего года к предыдущему: `{fmt_value((change.get('count') or {}).get('diff'))}` дел, `{fmt_money((change.get('claim_sum') or {}).get('diff'))}` требований.",
                f"- Давность последнего всплеска, месяцев: `{fmt_value(signal_context.get('last_burst_age_months'))}`",
                f"- Доля дел, где компания была истцом / ответчиком: `{fmt_value(signal_context.get('plaintiff_share'))}` / `{fmt_value(signal_context.get('defendant_share'))}`",
            ]
        )
    else:
        lines.extend(
            [
                "- Судебная история за выбранный lookback-период отсутствует.",
                "- Исторический судебный фон не рассчитывается: в выбранном периоде нет месяцев с делами, поэтому сравнивать текущую активность не с чем.",
            ]
        )

    lines.extend(["", "## Таймлайн сигнала"])
    if timeline.get("nonzero_rows"):
        lines.extend(
            [
                f"- Фокусная дата таймлайна: `{fmt_date(timeline.get('focus_end'))}`",
                f"- Глубина таймлайна: `{fmt_value(timeline.get('months_back'))}` мес.",
                "",
                "| Месяц | Во всплеске | Судебные дела | Сумма исков | ФССП | ЕФРСБ |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for item in timeline.get("nonzero_rows") or []:
            lines.append(
                f"| {item.get('month')} | {'да' if item.get('in_burst') else 'нет'} | {item.get('cases')} | {fmt_money(item.get('claim_sum'))} | {item.get('enforcement_count')} | {item.get('efrsb_count')} |"
            )
    else:
        lines.extend(
            [
                "- Существенных помесячных сигналов в выбранном окне не найдено.",
            ]
        )

    lines.extend(["", "## Масштаб судебной нагрузки относительно бизнеса"])
    if has_burst(burst):
        lines.extend(
            [
                f"- Требования к компании как к ответчику к уставному капиталу: `{fmt_value(scale.get('claim_to_capital'))}`",
                f"- Требования к компании как к ответчику к уплаченным налогам: `{fmt_value(scale.get('claim_to_tax_paid'))}`",
                f"- Требования к компании как к ответчику к активам: `{fmt_value(scale.get('claim_to_assets'))}`",
                f"- Требования к компании как к ответчику к выручке: `{fmt_value(scale.get('claim_to_revenue'))}`",
                f"- Дел, где компания ответчик, к численности сотрудников: `{fmt_value(scale.get('cases_to_headcount'))}`",
                f"- Дел, где компания ответчик, на 1 млн уплаченных налогов: `{fmt_value(scale.get('cases_to_tax_paid_mln'))}`",
                f"- Общая сумма судебных требований к уставному капиталу, включая дела где компания истец: `{fmt_value(scale.get('litigation_claim_to_capital'))}`",
            ]
        )
    else:
        lines.extend(
            [
                "- Масштабирование судебной нагрузки не применяется, потому что нет найденного судебного всплеска.",
                "- Нулевые court-ratio в JSON означают отсутствие судебной нагрузки, а не “нулевой риск компании”.",
            ]
        )

    lines.extend(
        [
            "",
            "## Сравнение с отраслью",
            f"- Группа ОКВЭД-2: `{industry.get('industry_okved2') or 'нет данных'}`",
            f"- Размер отраслевой выборки: `{industry.get('industry_sample_size')}`",
            f"- Подход к сравнению: `{baseline_scope_ru(industry.get('baseline_scope')) or 'нет данных'}`",
            f"- Размер базы для сравнения: `{industry.get('baseline_sample_size')}`",
            f"- Нагрузка по числу дел относительно высокого уровня в сопоставимой группе: `{fmt_value(industry.get('case_to_industry_p90'))}`",
            f"- Нагрузка по сумме исков относительно высокого уровня в сопоставимой группе: `{fmt_value(industry.get('claim_to_industry_p90'))}`",
            f"- Вывод по сравнению: `{'нагрузка выше высокого уровня в сопоставимой группе' if (industry.get('case_to_industry_p90') or 0) > 1 or (industry.get('claim_to_industry_p90') or 0) > 1 else 'нагрузка не выглядит экстремальной по сравнению с похожими компаниями'}`",
            "",
            "## Финансовый и долговой контекст",
            f"- Исполнительных производств до даты прогноза: `{fmt_value(context.get('enforcement_count'))}`",
            f"- Долг по исполнительным производствам до даты прогноза: {fmt_money(context.get('enforcement_debt'))}",
            f"- Сообщений ЕФРСБ до даты прогноза: `{fmt_value(context.get('efrsb_count'))}`",
            f"- Уставный капитал / профильный масштаб: {fmt_money(context.get('scale_capital') or context.get('authorized_capital'))}",
            f"- Среднесписочная численность: `{fmt_value(context.get('profile_headcount'))}`",
            f"- Лицензий в карточке: `{fmt_value(context.get('licenses_count'))}`",
            *(
                [
                    f"- Год финансового снимка: `{fmt_value(context.get('financial_snapshot_year'))}{' (устарел)' if context.get('financial_snapshot_stale') else ''}`",
                    f"- Активы: {fmt_money(context.get('assets'))}",
                    f"- Выручка: {fmt_money(context.get('revenue'))}",
                    f"- Чистая прибыль: {fmt_money(context.get('net_profit'))}",
                    f"- Собственный капитал: {fmt_money(context.get('equity'))}",
                ]
                if context.get("financial_snapshot_year")
                else ["- Стандартная финансовая отчетность в источнике не найдена; профильный масштаб выше сохраняется в отчете и используется моделью."]
            ),
            *(
                [
                    f"- Страховой контекст ЦБ: `{fmt_value((context.get('cbr_insurance') or {}).get('report_year'))}`, рег. номер `{(context.get('cbr_insurance') or {}).get('cbr_register_number')}`",
                    f"- Норматив капитала к обязательствам страховщика на {(context.get('cbr_insurance') or {}).get('ratio_date')}: `{fmt_value((context.get('cbr_insurance') or {}).get('capital_obligation_ratio'))}`",
                    f"- Страховые премии за год: {fmt_money((context.get('cbr_insurance') or {}).get('gross_premiums_total'))}",
                    f"- Страховые выплаты за год: {fmt_money((context.get('cbr_insurance') or {}).get('gross_payouts_total'))}",
                    f"- Источник страхового контекста: {(context.get('cbr_insurance') or {}).get('performance_url')}",
                ]
                if context.get("cbr_insurance")
                else []
            ),
            "",
            "## Контекст сигнала",
            "- Факторы ниже помогают интерпретировать, из чего складывается риск-профиль компании в отчете.",
            *([explain_factor(factor) for factor in (signal_context.get("factors") or [])] or ["- Контекстных факторов не выделено."]),
            "",
            "## Итоговая интерпретация",
        ]
    )
    lines.extend(assessment["interpretation"])
    lines.append("")
    return "\n".join(lines)

def assess_company(
    inn: str,
    *,
    force: bool,
    max_extra_case_pages: int,
    max_extra_enforcement_pages: int,
    court_lookback_years: int = DEFAULT_COURT_LOOKBACK_YEARS,
) -> dict[str, Any]:
    company_dir = collect_company(
        inn,
        force=force,
        max_extra_case_pages=max_extra_case_pages,
        max_extra_enforcement_pages=max_extra_enforcement_pages,
        court_lookback_years=court_lookback_years,
    )
    inputs = load_company_inputs(inn, company_dir, court_lookback_years=court_lookback_years)
    company_payload = inputs["raw"]
    data = inputs["data"]
    case_frame = inputs["case_frame"]
    enforcement_frame = inputs["enforcement_frame"]
    efrsb_frame = inputs["efrsb_frame"]
    finances = inputs["finances"]
    gate = status_gate(company_payload)
    row = build_scoring_row(inn, company_dir, court_lookback_years=court_lookback_years)
    scale = add_scale_normalized_features(row)
    industry = industry_baseline(row)
    court_history = court_history_context(inn, company_dir, court_lookback_years=court_lookback_years)

    service_metrics = json.loads(SERVICE_SUMMARY_FILE.read_text(encoding="utf-8"))
    threshold = (
        service_metrics.get("service_threshold")
        or (service_metrics.get("train_test_diagnostics") or {}).get("train_selected_threshold")
        or (service_metrics.get("oof_cv_metrics") or {}).get("threshold")
        or 0.5
    )
    threshold_name = service_metrics.get("service_threshold_name") or "train_selected_threshold"
    validation_metrics = (service_metrics.get("holdout_metrics_by_threshold") or {}).get(
        service_metrics.get("service_threshold_name") or "",
        {},
    )
    if not validation_metrics:
        validation_metrics = (service_metrics.get("train_test_diagnostics") or {}).get("test_metrics", {})
    cases_meta = (read_json(company_dir / "legal_cases_pages.json").get("meta") or {})
    legal_cases_truncated = bool(cases_meta.get("truncated_by_max_pages"))
    probability = None
    if gate["mode"] == "predictive" and int(row.get("last_burst_has_burst") or 0) == 1 and not legal_cases_truncated:
        # Вероятность считаю только для действующих компаний и полной судебной истории; так меньше ложной уверенности.
        probability = score_predictive(row, load_service_columns())
    signal_context = build_signal_context(probability, row, scale, court_history)
    service_probability = probability

    score = {
        "probability": service_probability,
        "model_probability": probability,
        "threshold": threshold,
        "risk_level": risk_level(service_probability, threshold),
        "target": service_metrics.get("target") or "failed_within_12m_from_anchor",
        "model": service_metrics.get("model_name") or "risk_model_12m",
        "prediction_date": fmt_date(row.get("service_prediction_date")),
    }
    last_burst = {
        "start_date": str(row.get("last_burst_start_date")),
        "end_date": str(row.get("last_burst_end_date")),
        "months": row.get("last_burst_months"),
        "total_cases": row.get("last_burst_total_cases"),
        "total_claim_sum": row.get("last_burst_total_claim_sum"),
        "cases_per_month": row.get("last_burst_cases_per_month"),
        "claim_per_month": row.get("last_burst_claim_per_month"),
        "defendant_cases": row.get("last_burst_defendant_cases"),
        "plaintiff_cases": row.get("last_burst_plaintiff_cases"),
    }
    financial_snapshot_year = bem.safe_float(row.get("last_anchor_fin_snapshot_year"))
    financial_snapshot_stale = is_financial_snapshot_stale(financial_snapshot_year, row.get("service_prediction_date"))
    financial_context_available = financial_snapshot_year is not None and not financial_snapshot_stale
    cbr_insurance = read_json(company_dir / "cbr_insurance_context.json")
    if cbr_insurance.get("status") != "ok":
        cbr_insurance = None
    context = {
        "enforcement_count": row.get("last_anchor_enf_count_total"),
        "enforcement_debt": row.get("last_anchor_enf_debt_total"),
        "efrsb_count": row.get("last_anchor_efrsb_total_count"),
        "scale_capital": bem.safe_float(row.get("profile_capital_sum")) or bem.safe_float(row.get("authorized_capital")),
        "authorized_capital": bem.safe_float(row.get("authorized_capital")),
        "profile_headcount": bem.safe_float(row.get("profile_headcount")),
        "licenses_count": bem.safe_float(row.get("raw_licenses_count")),
        "branches_count": bem.safe_float(row.get("raw_branch_count")),
        "financial_snapshot_year": int(financial_snapshot_year) if financial_snapshot_year is not None else None,
        "financial_snapshot_stale": financial_snapshot_stale,
        "assets": bem.safe_float(row.get("last_anchor_fin_assets_t0")) if financial_context_available else None,
        "revenue": bem.safe_float(row.get("last_anchor_fin_revenue_t0")) if financial_context_available else None,
        "net_profit": bem.safe_float(row.get("last_anchor_fin_net_profit_t0")) if financial_context_available else None,
        "equity": bem.safe_float(row.get("last_anchor_fin_equity_t0")) if financial_context_available else None,
        "cbr_insurance": cbr_insurance,
    }
    data_quality = build_data_quality(company_dir, row, last_burst, court_history, industry)
    result_state = build_result_state(gate, score, data_quality, last_burst, context)
    source_freshness = build_source_freshness(
        company_dir,
        court_lookback_years=court_lookback_years,
        case_frame=case_frame,
        enforcement_frame=enforcement_frame,
        efrsb_frame=efrsb_frame,
        finances=finances,
    )
    timeline = build_timeline(case_frame, enforcement_frame, efrsb_frame, last_burst)

    interpretation = []
    if gate["mode"] == "known_negative":
        interpretation.append("- Компания уже имеет негативный статус в Ofdata. Отчет следует читать как ретроспективный, а не как прогноз.")
    elif probability is None:
        interpretation.append("- Прогноз не рассчитан из-за неопределенного статуса компании.")
    else:
        interpretation.append("- Прогноз рассчитан только после проверки статуса, потому что компания выглядит действующей.")
        interpretation.append(f"- Расчет выполнен моделью `{service_metrics.get('model_name') or 'risk_model_12m'}` для прогноза на 12 месяцев.")
        if (scale.get("claim_to_capital") or 0) > 1 and (row.get("last_burst_defendant_cases") or 0) > 0:
            interpretation.append("- Требования к компании как к ответчику превышают уставный капитал; это усиливает стресс-сигнал.")
        if (industry.get("case_to_industry_p90") or 0) > 1:
            interpretation.append("- Судебная нагрузка выше 90-го перцентиля по отраслевой группе в обучающей выборке.")
        if (row.get("last_anchor_efrsb_total_count") or 0) > 0:
            interpretation.append("- Есть сообщения ЕФРСБ до даты прогноза; это процедурный банкротный контекст.")
        if (row.get("last_anchor_enf_count_total") or 0) > 0:
            interpretation.append("- Есть исполнительные производства до даты прогноза; это долговой контекст.")
        if court_history.get("continuous_litigation_baseline"):
            ratio_12m = ((court_history.get("windows") or {}).get("12m") or {}).get("case_activity_ratio")
            if ratio_12m is not None and ratio_12m < 1.2:
                interpretation.append("- Судебная активность выглядит как длинный операционный фон: последние 12 месяцев не выше исторической нормы по числу дел.")
            else:
                interpretation.append("- У компании есть длинная судебная история; абсолютное число дел нужно оценивать относительно ее собственного обычного фона, а не само по себе.")
        if not interpretation[-1].startswith("- Есть") and not interpretation[-1].startswith("- Судебная") and not interpretation[-1].startswith("- Сумма"):
            interpretation.append("- Основной сигнал формируется из сочетания последнего судебного всплеска и финансового контекста.")

    executive_summary = build_executive_summary(gate, result_state, score, data_quality, last_burst, context, industry, scale)

    assessment = {
        "company": {
            "inn": inn,
            "name": data.get("НаимСокр") or data.get("НаимПолн"),
            "ogrn": data.get("ОГРН"),
            "okved": (data.get("ОКВЭД") or {}).get("Код"),
            "okved_name": (data.get("ОКВЭД") or {}).get("Наим"),
        },
        "status_gate": gate,
        "score": score,
        "result_state": result_state,
        "data_quality": data_quality,
        "source_freshness": source_freshness,
        "signal_context": signal_context,
        "model_validation": {
            "rows": service_metrics.get("rows"),
            "feature_count": service_metrics.get("feature_count"),
            "positive_rate": service_metrics.get("positive_rate"),
            "roc_auc": validation_metrics.get("roc_auc"),
            "pr_auc": validation_metrics.get("pr_auc"),
            "f1": validation_metrics.get("f1"),
            "accuracy": validation_metrics.get("accuracy"),
            "recall": validation_metrics.get("recall"),
            "precision": validation_metrics.get("precision"),
            "service_threshold_name": threshold_name,
            "service_threshold": threshold,
            "target": service_metrics.get("target"),
        },
        "last_burst": last_burst,
        "court_history": court_history,
        "court_windows": court_window_summary(row),
        "timeline": timeline,
        "scale_normalization": scale,
        "industry_baseline": industry,
        "context": context,
        "executive_summary": executive_summary,
        "interpretation": interpretation,
    }
    return assessment

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an Ofdata risk assessment report for a company.")
    parser.add_argument("inn_positional", nargs="?", help="Company INN. Kept for short CLI calls.")
    parser.add_argument("--inn", dest="inn_option", help="Company INN.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--court-lookback-years", type=int, default=DEFAULT_COURT_LOOKBACK_YEARS)
    parser.add_argument("--max-extra-case-pages", type=int, default=DEFAULT_MAX_EXTRA_CASE_PAGES)
    parser.add_argument("--max-extra-enforcement-pages", type=int, default=DEFAULT_MAX_EXTRA_ENFORCEMENT_PAGES)
    args = parser.parse_args()
    args.inn = args.inn_option or args.inn_positional
    if not args.inn:
        parser.error("provide INN either as positional argument or --inn")
    return args

def main() -> None:
    args = parse_args()
    assessment = assess_company(
        args.inn,
        force=args.force,
        max_extra_case_pages=args.max_extra_case_pages,
        max_extra_enforcement_pages=args.max_extra_enforcement_pages,
        court_lookback_years=args.court_lookback_years,
    )
    slug = report_slug(args.inn, assessment["company"].get("name"))
    report_dir = REPORTS_DIR / slug
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / f"{slug}.json", assessment)
    report_path = report_dir / f"{slug}.md"
    report_path.write_text(build_report(assessment), encoding="utf-8")
    print(short_summary(assessment, report_path))

if __name__ == "__main__":
    main()
