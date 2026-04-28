from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
ENGINE_DIR = ROOT_DIR / "recovery_index" / "service"
# Короткий CLI-ярлык: на защите так проще показать запуск, без длинного пути до engine.
sys.path.insert(0, str(ENGINE_DIR))

from risk_engine import (  # noqa: E402
    DEFAULT_COURT_LOOKBACK_YEARS,
    DEFAULT_MAX_EXTRA_CASE_PAGES,
    DEFAULT_MAX_EXTRA_ENFORCEMENT_PAGES,
    REPORTS_DIR,
    assess_company,
    build_report,
    report_slug,
    short_summary,
    write_json,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assess company risk by INN.")
    parser.add_argument("inn_positional", nargs="?", help="Company INN.")
    parser.add_argument("--inn", dest="inn_option", help="Company INN.")
    parser.add_argument("--force", action="store_true", help="Refresh Ofdata cache.")
    parser.add_argument("--court-lookback-years", type=int, default=DEFAULT_COURT_LOOKBACK_YEARS)
    parser.add_argument("--max-extra-case-pages", type=int, default=DEFAULT_MAX_EXTRA_CASE_PAGES)
    parser.add_argument("--max-extra-enforcement-pages", type=int, default=DEFAULT_MAX_EXTRA_ENFORCEMENT_PAGES)
    args = parser.parse_args()
    args.inn = "".join(ch for ch in (args.inn_option or args.inn_positional or "") if ch.isdigit())
    if len(args.inn) not in {10, 12}:
        parser.error("provide a valid 10- or 12-digit INN, e.g. python3 assess.py 7715829230")
    return args

def years_label(value: int) -> str:
    rem10 = value % 10
    rem100 = value % 100
    if rem10 == 1 and rem100 != 11:
        return "год"
    if rem10 in {2, 3, 4} and rem100 not in {12, 13, 14}:
        return "года"
    return "лет"

def main() -> None:
    args = parse_args()
    print("")
    print("Собираю данные Ofdata и считаю риск...")
    print(f"ИНН: {args.inn}")
    print(f"Суды: последние {args.court_lookback_years} {years_label(args.court_lookback_years)}")
    print("")

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
    # Пишу и JSON, и markdown: первый удобен для проверки чисел, второй - для чтения глазами.
    write_json(report_dir / f"{slug}.json", assessment)
    report_path = report_dir / f"{slug}.md"
    report_path.write_text(build_report(assessment), encoding="utf-8")
    print(short_summary(assessment, report_path))

if __name__ == "__main__":
    main()
