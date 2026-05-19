from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from api_client import get_json

ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT_DIR / "recovery_index"
SEED_FILE = PROJECT_DIR / "data" / "companies_seed.csv"
RAW_DIR = PROJECT_DIR / "service" / "raw"
DEFAULT_CASE_LOOKBACK_YEARS = 3
DEFAULT_CASE_PAGE_LIMIT = 200
DEFAULT_ENFORCEMENT_PAGE_LIMIT = 3

def safe_filename(value: str) -> str:
    return re.sub(r"[^0-9A-Za-zА-Яа-я._-]+", "_", str(value)).strip("_") or "item"

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def collect_pages(endpoint: str, *, page_limit: int, stop_before: pd.Timestamp | None = None, **params: Any) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    stopped_by_cutoff = False
    for page in range(1, page_limit + 1):
        payload = get_json(endpoint, page=page, limit=100, **params)
        pages.append(payload)
        items = (payload.get("data") or {}).get("Записи") or []
        if not items:
            break
        if stop_before is not None:
            item_dates = [pd.to_datetime(item.get("Дата") or item.get("ИспПрДата"), errors="coerce") for item in items]
            if any(pd.notna(item_date) and item_date < stop_before for item_date in item_dates):
                stopped_by_cutoff = True
                break
        total_pages = (payload.get("data") or {}).get("СтрВсего")
        if total_pages and page >= int(total_pages):
            break
    return {"pages": pages, "stopped_by_cutoff": stopped_by_cutoff, "page_limit": page_limit}

def normalize_seed(seed: pd.DataFrame) -> pd.DataFrame:
    frame = seed.copy()
    if "company_inn" not in frame.columns:
        if "inn" in frame.columns:
            frame["company_inn"] = frame["inn"]
        else:
            raise ValueError("Seed file must contain 'company_inn' or 'inn'.")
    frame["company_inn"] = frame["company_inn"].astype(str).str.extract(r"(\d+)")[0].str.zfill(10)
    frame = frame.dropna(subset=["company_inn"]).drop_duplicates(subset=["company_inn"]).reset_index(drop=True)
    return frame

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified final Ofdata collector for the balanced company seed.")
    parser.add_argument("--seed-file", default=str(SEED_FILE))
    parser.add_argument("--court-lookback-years", type=int, default=DEFAULT_CASE_LOOKBACK_YEARS)
    parser.add_argument("--case-page-limit", type=int, default=DEFAULT_CASE_PAGE_LIMIT)
    parser.add_argument("--enforcement-page-limit", type=int, default=DEFAULT_ENFORCEMENT_PAGE_LIMIT)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--max-companies", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seed = normalize_seed(pd.read_csv(args.seed_file, dtype="string"))
    if args.max_companies is not None:
        seed = seed.iloc[args.start_from : args.start_from + args.max_companies].reset_index(drop=True)
    else:
        seed = seed.iloc[args.start_from :].reset_index(drop=True)

    observation_end = pd.Timestamp("2026-04-23")
    cutoff = observation_end - pd.DateOffset(years=args.court_lookback_years)

    for index, row in enumerate(seed.to_dict("records"), start=1):
        inn = row["company_inn"]
        company_dir = RAW_DIR / safe_filename(inn)
        company_dir.mkdir(parents=True, exist_ok=True)
        print(f"[collect-final] {index}/{len(seed)} INN={inn}", flush=True)

        company_file = company_dir / "company.json"
        finances_file = company_dir / "finances.json"
        cases_file = company_dir / "legal_cases_pages.json"
        enforcements_file = company_dir / "enforcements_pages.json"

        if args.force or not company_file.exists():
            write_json(company_file, get_json("company", inn=inn))
        if args.force or not finances_file.exists():
            write_json(finances_file, get_json("finances", inn=inn))
        if args.force or not cases_file.exists():
            write_json(
                cases_file,
                collect_pages("legal-cases", inn=inn, sort="-date", page_limit=args.case_page_limit, stop_before=cutoff),
            )
        if args.force or not enforcements_file.exists():
            write_json(
                enforcements_file,
                collect_pages("enforcements", inn=inn, sort="-date", page_limit=args.enforcement_page_limit),
            )

if __name__ == "__main__":
    main()
