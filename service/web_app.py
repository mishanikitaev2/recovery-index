from __future__ import annotations

import json
import math
from datetime import date, datetime
from typing import Any

from flask import Flask, jsonify, redirect, render_template_string, request, url_for

from risk_engine import (
    DEFAULT_COURT_LOOKBACK_YEARS,
    DEFAULT_MAX_EXTRA_CASE_PAGES,
    DEFAULT_MAX_EXTRA_ENFORCEMENT_PAGES,
    REPORTS_DIR,
    assess_company,
    build_report,
    gate_mode_ru,
    report_slug,
    risk_level_ru,
    target_ru,
    write_json,
)

app = Flask(__name__)
EMPTY_DATES = {"", "NaT", "None", "nan"}

def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on", "да"}

def assess_from_request(inn: str) -> dict[str, Any]:
    force = as_bool(request.args.get("force"))
    court_lookback_years = int(request.args.get("court_lookback_years") or DEFAULT_COURT_LOOKBACK_YEARS)
    max_extra_case_pages = int(request.args.get("max_extra_case_pages") or DEFAULT_MAX_EXTRA_CASE_PAGES)
    max_extra_enforcement_pages = int(request.args.get("max_extra_enforcement_pages") or DEFAULT_MAX_EXTRA_ENFORCEMENT_PAGES)

    assessment = assess_company(
        inn,
        force=force,
        max_extra_case_pages=max_extra_case_pages,
        max_extra_enforcement_pages=max_extra_enforcement_pages,
        court_lookback_years=court_lookback_years,
    )
    slug = report_slug(inn, assessment["company"].get("name"))
    report_dir = REPORTS_DIR / slug
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{slug}.json"
    markdown_path = report_dir / f"{slug}.md"
    assessment["report_file"] = str(markdown_path)
    write_json(json_path, clean_json(assessment))
    markdown_path.write_text(build_report(assessment), encoding="utf-8")
    return clean_json(assessment)

def fmt_money(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return "нет данных"
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f} млрд ₽"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.2f} млн ₽"
    return f"{number:,.0f} ₽".replace(",", " ")

def fmt_num(value: Any, digits: int = 3) -> str:
    number = to_float(value)
    if number is None:
        return "нет данных" if value is None else str(value)
    if number.is_integer():
        return str(int(number))
    if abs(number) >= 10:
        return f"{number:.1f}".rstrip("0").rstrip(".")
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")

def fmt_probability(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return "нет данных"
    return f"{number:.1%}"

def fmt_date(value: Any) -> str:
    if value is None:
        return "нет данных"
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value)
    if text in EMPTY_DATES:
        return "нет данных"
    try:
        return datetime.fromisoformat(text[:10]).date().isoformat()
    except ValueError:
        return text

def has_burst(burst: Any) -> bool:
    if not isinstance(burst, dict):
        return False
    cases = to_float(burst.get("defendant_cases") or burst.get("total_cases")) or 0.0
    start = str(burst.get("start_date") or "")
    end = str(burst.get("end_date") or "")
    return cases > 0 and start not in EMPTY_DATES and end not in EMPTY_DATES

def has_court_history(court: Any) -> bool:
    if not isinstance(court, dict):
        return False
    return (to_float(court.get("defendant_cases_fetched")) or 0.0) > 0

PAGE_TEMPLATE = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Recovery Index · Ofdata Assessment</title>
  <style>
    :root {
      --ink: #17201b;
      --muted: #68746d;
      --paper: #fbf7ee;
      --card: #fffdf7;
      --line: #e4dccd;
      --good: #22724d;
      --watch: #b66a00;
      --bad: #a9382d;
      --blue: #215f8f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 20% 10%, rgba(244, 206, 122, .30), transparent 35%),
        linear-gradient(135deg, #fbf7ee 0%, #eef4ef 100%);
      min-height: 100vh;
    }
    header {
      padding: 36px 5vw 22px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 253, 247, .70);
      backdrop-filter: blur(10px);
    }
    h1 { margin: 0 0 8px; font-size: clamp(30px, 5vw, 58px); letter-spacing: -1.5px; }
    .subtitle { color: var(--muted); font-size: 18px; max-width: 920px; line-height: 1.45; }
    main { padding: 28px 5vw 60px; }
    form {
      display: grid;
      grid-template-columns: minmax(180px, 320px) repeat(3, auto) auto;
      gap: 12px;
      align-items: center;
      margin-bottom: 24px;
    }
    input[type=text] {
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--card);
      font-size: 18px;
    }
    input.short-number {
      width: 72px;
      padding: 8px;
      border-radius: 10px;
      border: 1px solid var(--line);
    }
    label { color: var(--muted); font-size: 15px; white-space: nowrap; }
    button, .button {
      border: 0;
      border-radius: 14px;
      padding: 14px 18px;
      background: var(--ink);
      color: white;
      font-weight: 700;
      cursor: pointer;
      text-decoration: none;
      display: inline-block;
      font-family: inherit;
      font-size: 16px;
    }
    .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 16px; }
    .card {
      grid-column: span 4;
      background: rgba(255, 253, 247, .86);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 20px;
      box-shadow: 0 18px 45px rgba(45, 37, 23, .07);
    }
    .wide { grid-column: span 8; }
    .full { grid-column: 1 / -1; }
    h2 { margin: 0 0 14px; font-size: 22px; }
    .metric { font-size: 34px; font-weight: 800; line-height: 1; letter-spacing: -1px; }
    .muted { color: var(--muted); }
    .pill {
      display: inline-flex;
      padding: 7px 11px;
      border-radius: 999px;
      background: #eee7d8;
      color: var(--ink);
      font-size: 14px;
      font-weight: 700;
      margin: 3px 4px 3px 0;
    }
    .pill.good { background: #dceee5; color: var(--good); }
    .pill.watch { background: #f6e6c9; color: var(--watch); }
    .pill.bad { background: #f4d8d4; color: var(--bad); }
    .kv { display: grid; grid-template-columns: 1.1fr 1fr; gap: 8px 16px; font-size: 15px; }
    .kv div:nth-child(odd) { color: var(--muted); }
    .interpretation li { margin: 8px 0; line-height: 1.45; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }
    th { color: var(--muted); font-weight: 700; }
    .table-wrap {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255, 253, 247, .55);
    }
    pre {
      white-space: pre-wrap;
      overflow: auto;
      background: #17201b;
      color: #f7f1e4;
      border-radius: 18px;
      padding: 18px;
      max-height: 520px;
    }
    @media (max-width: 900px) {
      form { grid-template-columns: 1fr; }
      .card, .wide { grid-column: 1 / -1; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Recovery Index</h1>
    <div class="subtitle">Отчет по компании на основе статуса, судебной активности, финансов, ФССП, ЕФРСБ и сервисной риск-модели.</div>
  </header>
  <main>
    <form action="/assess" method="get">
      <input type="text" name="inn" placeholder="ИНН компании" value="{{ inn or '' }}" required>
      <label>суды за <input class="short-number" type="number" name="court_lookback_years" value="{{ court_lookback_years or 2 }}" min="1" max="10"> лет</label>
      <label><input type="checkbox" name="force" value="1" {% if force %}checked{% endif %}> обновить данные</label>
      <button type="submit">Оценить</button>
    </form>

    {% if error %}
      <div class="card full"><h2>Ошибка</h2><p>{{ error }}</p></div>
    {% endif %}

    {% if assessment %}
      {% set gate = assessment.status_gate %}
      {% set score = assessment.score %}
      {% set state = assessment.result_state %}
      {% set dq = assessment.data_quality %}
      {% set executive = assessment.executive_summary %}
      {% set burst = assessment.last_burst %}
      {% set court = assessment.court_history %}
      {% set windows = assessment.court_windows or {} %}
      {% set ctx = assessment.context %}
      {% set timeline = assessment.timeline %}
      {% set freshness = assessment.source_freshness or {} %}
      <section class="grid">
        <div class="card wide">
          <h2>{{ assessment.company.name or assessment.company.inn }}</h2>
          <span class="pill {% if gate.mode == 'predictive' %}good{% elif gate.mode == 'known_negative' %}bad{% else %}watch{% endif %}">{{ gate_mode_ru(gate.mode) }}</span>
          <span class="pill">{{ gate.status_name or 'нет статуса' }}</span>
          <span class="pill">ИНН {{ assessment.company.inn }}</span>
          <span class="pill {% if state.code == 'risk_detected' %}bad{% elif state.code == 'no_material_risk' %}good{% else %}watch{% endif %}">{{ state.label or 'нет состояния' }}</span>
          <p class="muted">ОКВЭД: {{ assessment.company.okved or 'нет данных' }} · {{ assessment.company.okved_name or '' }}</p>
        </div>
        <div class="card">
          <h2>Итоговая оценка риска</h2>
          <div class="metric">{{ fmt_probability(score.probability) if score.probability is not none else 'N/A' }}</div>
          <p><span class="pill {% if score.risk_level == 'high' %}bad{% elif score.risk_level == 'watch' %}watch{% else %}good{% endif %}">{{ risk_level_ru(score.risk_level) }}</span></p>
          <p class="muted">модель: {{ score.model or 'risk_model_12m' }}<br>что прогнозируем: {{ target_ru(score.target) }}<br>порог риска: {{ fmt_probability(score.threshold) }}</p>
        </div>
        <div class="card">
          <h2>Качество данных</h2>
          <div class="metric">{{ fmt_num(dq.overall_percent, 0) }}%</div>
          <p><span class="pill {% if dq.confidence_level == 'high' %}good{% elif dq.confidence_level == 'medium' %}watch{% else %}bad{% endif %}">{{ dq.confidence_label or 'нет данных' }}</span></p>
          <p class="muted">источники: {{ fmt_num(dq.source_coverage_percent, 0) }}%<br>аналитические блоки модели: {{ fmt_num(dq.analytic_coverage_percent, 0) }}%<br>ограничения модели: {{ '; '.join(dq.notes) if dq.notes else 'нет' }}</p>
        </div>
        <div class="card full">
          <h2>Ключевые выводы</h2>
          <ul class="interpretation">
            {% for item in executive %}
              <li>{{ item }}</li>
            {% endfor %}
          </ul>
        </div>
        <div class="card">
          <h2>Последний судебный всплеск</h2>
          {% if has_burst(burst) %}
            <div class="kv">
              <div>Период</div><div>{{ fmt_date(burst.start_date) }} - {{ fmt_date(burst.end_date) }}</div>
              <div>Дел как ответчик</div><div>{{ fmt_num(burst.defendant_cases, 0) }}</div>
              <div>Сумма требований к компании</div><div>{{ fmt_money(burst.defendant_claim_sum) }}</div>
              <div>Дел ответчика/мес</div><div>{{ fmt_num(burst.defendant_cases_per_month) }}</div>
              <div>Истцовых дел в том же эпизоде</div><div>{{ fmt_num(burst.plaintiff_cases, 0) }}</div>
            </div>
          {% else %}
            <p class="muted">Судебных дел за выбранный период не найдено, поэтому судебный всплеск не сформирован.</p>
          {% endif %}
        </div>
        <div class="card">
          <h2>Исторический судебный фон</h2>
          {% if has_court_history(court) %}
            <div class="kv">
              <div>Найдено дел всего</div><div>{{ fmt_num(court.total_cases_fetched, 0) }}</div>
              <div>Из них как ответчик</div><div>{{ fmt_num(court.defendant_cases_fetched, 0) }}</div>
              {% if freshness.legal_cases.truncated_by_max_pages %}
                <div>Ограничение</div><div>загружено {{ freshness.legal_cases.pages_fetched }} из {{ freshness.legal_cases.total_pages_reported }} страниц</div>
              {% endif %}
              <div>Период</div><div>{{ fmt_date(court.first_case_date) }} - {{ fmt_date(court.last_case_date) }}</div>
              <div>Активные месяцы ответчика</div><div>{{ court.active_months }} / {{ court.span_months }}</div>
              <div>Есть устойчивый фон</div><div>{{ 'да' if court.continuous_litigation_baseline else 'нет' }}</div>
              <div>Активность дел за 12 месяцев к своей истории</div><div>{{ fmt_num(court.windows.get('12m', {}).get('case_activity_ratio')) }}</div>
              <div>Последние 12 месяцев</div><div>{{ fmt_num(windows.get('last12', {}).get('defendant_count'), 0) }} дел · {{ fmt_money(windows.get('last12', {}).get('defendant_claim_sum')) }}</div>
              <div>Предыдущие 12 месяцев</div><div>{{ fmt_num(windows.get('previous12', {}).get('defendant_count'), 0) }} дел · {{ fmt_money(windows.get('previous12', {}).get('defendant_claim_sum')) }}</div>
              <div>Все 24 месяца</div><div>{{ fmt_num(windows.get('last24', {}).get('defendant_count'), 0) }} дел · {{ fmt_money(windows.get('last24', {}).get('defendant_claim_sum')) }}</div>
              <div>Изменение год к году</div><div>{{ fmt_num(windows.get('change', {}).get('defendant_count', {}).get('diff'), 0) }} дел · {{ fmt_money(windows.get('change', {}).get('defendant_claim_sum', {}).get('diff')) }}</div>
            </div>
          {% else %}
            <p class="muted">Судебная история за выбранный период отсутствует, поэтому сравнение с обычным фоном компании не рассчитывается.</p>
          {% endif %}
        </div>
        <div class="card">
          <h2>Финансовый и масштабный контекст</h2>
          <div class="kv">
            <div>Уставный капитал</div><div>{{ fmt_money(ctx.scale_capital or ctx.authorized_capital) }}</div>
            <div>Среднесписочная численность</div><div>{{ fmt_num(ctx.profile_headcount, 0) }}</div>
            <div>Категория МСП</div><div>{{ ctx.profile_msp_category or 'нет данных' }}</div>
          </div>
          {% if ctx.financial_snapshot_year %}
          <div class="kv">
            <div>Год финансового снимка</div><div>{{ ctx.financial_snapshot_year or 'нет' }}{% if ctx.financial_snapshot_stale %} (устарел){% endif %}</div>
            <div>Активы</div><div>{{ fmt_money(ctx.assets) }}</div>
            <div>Выручка</div><div>{{ fmt_money(ctx.revenue) }}</div>
            <div>Чистая прибыль</div><div>{{ fmt_money(ctx.net_profit) }}</div>
            <div>Собственный капитал</div><div>{{ fmt_money(ctx.equity) }}</div>
          </div>
          {% else %}
            <p class="muted">Стандартная финансовая отчетность в источнике не найдена. Профильный масштаб выше сохраняется в отчете и используется моделью.</p>
          {% endif %}
        </div>
        <div class="card">
          <h2>Долговой и банкротный контекст</h2>
          <div class="kv">
            <div>Исполнительных производств</div><div>{{ fmt_num(ctx.enforcement_count, 0) }}</div>
            <div>Долг по ФССП</div><div>{{ fmt_money(ctx.enforcement_debt) }}</div>
            <div>Сообщений ЕФРСБ</div><div>{{ fmt_num(ctx.efrsb_count, 0) }}</div>
          </div>
        </div>
        <div class="card full">
          <h2>Таймлайн сигнала</h2>
          {% if timeline.nonzero_rows %}
            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Месяц</th>
                    <th>Во всплеске</th>
                    <th>Ответчик</th>
                    <th>Истец</th>
                    <th>Сумма требований к компании</th>
                    <th>ФССП</th>
                    <th>ЕФРСБ</th>
                  </tr>
                </thead>
                <tbody>
                  {% for item in timeline.nonzero_rows %}
                    <tr>
                      <td>{{ item.month }}</td>
                      <td>{{ 'да' if item.in_burst else 'нет' }}</td>
                      <td>{{ item.defendant_cases }}</td>
                      <td>{{ item.plaintiff_cases }}</td>
                      <td>{{ fmt_money(item.defendant_claim_sum) }}</td>
                      <td>{{ item.enforcement_count }}</td>
                      <td>{{ item.efrsb_count }}</td>
                    </tr>
                  {% endfor %}
                </tbody>
              </table>
            </div>
          {% else %}
            <p class="muted">В доступном окне наблюдения не найдено помесячных сигналов для показа таймлайна.</p>
          {% endif %}
        </div>
        <div class="card full">
          <h2>Пояснение результата</h2>
          <ul class="interpretation">
            {% for item in assessment.interpretation %}
              <li>{{ item[2:] if item.startswith('- ') else item }}</li>
            {% endfor %}
          </ul>
        </div>
        <div class="card full">
          <h2>Технический JSON</h2>
          <pre>{{ assessment_json }}</pre>
        </div>
      </section>
    {% endif %}
  </main>
</body>
</html>
"""

@app.context_processor
def inject_formatters() -> dict[str, Any]:
    return {
        "fmt_money": fmt_money,
        "fmt_num": fmt_num,
        "fmt_probability": fmt_probability,
        "fmt_date": fmt_date,
        "has_burst": has_burst,
        "has_court_history": has_court_history,
        "gate_mode_ru": gate_mode_ru,
        "risk_level_ru": risk_level_ru,
        "target_ru": target_ru,
    }

@app.get("/")
def index() -> str:
    return render_template_string(PAGE_TEMPLATE, assessment=None, error=None, inn="", court_lookback_years=DEFAULT_COURT_LOOKBACK_YEARS, force=False)

@app.get("/assess")
def assess_query() -> Any:
    inn = "".join(ch for ch in (request.args.get("inn") or "") if ch.isdigit())
    if not inn:
        return render_template_string(PAGE_TEMPLATE, assessment=None, error="Укажи ИНН.", inn="", court_lookback_years=DEFAULT_COURT_LOOKBACK_YEARS, force=False)
    args = request.args.to_dict(flat=True)
    args.pop("inn", None)
    return redirect(url_for("assess_page", inn=inn, **args))

@app.get("/assess/<inn>")
def assess_page(inn: str) -> str:
    court_lookback_years = int(request.args.get("court_lookback_years") or DEFAULT_COURT_LOOKBACK_YEARS)
    force = as_bool(request.args.get("force"))
    try:
        assessment = assess_from_request(inn)
        error = None
    except Exception as exc:
        assessment = None
        error = str(exc)
    return render_template_string(
        PAGE_TEMPLATE,
        assessment=assessment,
        assessment_json=json.dumps(assessment, ensure_ascii=False, indent=2) if assessment else "",
        error=error,
        inn=inn,
        court_lookback_years=court_lookback_years,
        force=force,
    )

@app.get("/api/assess/<inn>")
def assess_api(inn: str) -> Any:
    try:
        return jsonify(assess_from_request(inn))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False, use_reloader=False, threaded=True)
