from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from http.client import RemoteDisconnected
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT_DIR / ".env"
BASE_URL = "https://api.ofdata.ru/v2"
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("OFDATA_TIMEOUT_SECONDS", "15"))
DEFAULT_RETRIES = int(os.environ.get("OFDATA_RETRIES", "2"))

def load_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result

def get_api_key() -> str:
    env = load_env_file()
    key = os.environ.get("OFDATA_API_KEY") or env.get("OFDATA_API_KEY")
    if not key:
        raise RuntimeError("OFDATA_API_KEY is missing")
    return key

def build_url(endpoint: str, **params: Any) -> str:
    query = {"key": get_api_key(), **{k: v for k, v in params.items() if v is not None}}
    return f"{BASE_URL}/{endpoint}?{urlencode(query)}"

def get_json(endpoint: str, **params: Any) -> dict[str, Any]:
    url = build_url(endpoint, **params)
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "rec-ind/ofdata-test"})
    last_error: Exception | None = None
    for attempt in range(DEFAULT_RETRIES):
        try:
            with urlopen(req, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except (RemoteDisconnected, TimeoutError, URLError) as exc:
            last_error = exc
            if attempt == DEFAULT_RETRIES - 1:
                break
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Ofdata request failed after retries: endpoint={endpoint}") from last_error

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Small Ofdata API probe client.")
    parser.add_argument("endpoint", help="Ofdata endpoint, e.g. company or finances")
    parser.add_argument("--inn", default=None)
    parser.add_argument("--ogrn", default=None)
    parser.add_argument("--limit", default=None)
    parser.add_argument("--page", default=None)
    parser.add_argument("--extended", default=None)
    args = parser.parse_args()
    data = get_json(
        args.endpoint,
        inn=args.inn,
        ogrn=args.ogrn,
        limit=args.limit,
        page=args.page,
        extended=args.extended,
    )
    print(json.dumps(data, ensure_ascii=False, indent=2))
