"""
auth.py — Bitunix API authentication

Bitunix uses a double SHA256 signature scheme:
  digest = sha256_hex(nonce + timestamp + api_key + query_params + body)
  sign   = sha256_hex(digest + secret_key)

Query params: sorted alphabetically by key, values only concatenated (no delimiters).
Timestamp:    milliseconds UTC (REST); seconds UTC (WebSocket — handled separately).
Nonce:        random 8-char alphanumeric string, regenerated per request.
"""

import hashlib
import random
import string
import time
import requests
from typing import Any

BASE_URL = "https://fapi.bitunix.com"


def _nonce() -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=8))


def _timestamp_ms() -> str:
    return str(int(time.time() * 1000))


def _serialize_params(params: dict) -> str:
    """Sort params by key, concatenate values only (Bitunix convention)."""
    if not params:
        return ""
    return "".join(str(v) for k, v in sorted(params.items()))


def _sign(api_key: str, secret_key: str, nonce: str, timestamp: str,
          query_params: dict, body: str) -> str:
    param_str = _serialize_params(query_params)
    digest = hashlib.sha256(
        (nonce + timestamp + api_key + param_str + body).encode()
    ).hexdigest()
    sign = hashlib.sha256(
        (digest + secret_key).encode()
    ).hexdigest()
    return sign


def _headers(api_key: str, secret_key: str,
             query_params: dict = None, body: str = "") -> dict:
    nonce     = _nonce()
    timestamp = _timestamp_ms()
    sign      = _sign(api_key, secret_key, nonce, timestamp,
                      query_params or {}, body)
    return {
        "api-key":      api_key,
        "nonce":        nonce,
        "timestamp":    timestamp,
        "sign":         sign,
        "Content-Type": "application/json",
    }


class BitunixClient:
    """Thin authenticated REST client for the Bitunix Futures API."""

    def __init__(self, api_key: str, secret_key: str):
        self.api_key    = api_key
        self.secret_key = secret_key
        self.session    = requests.Session()

    def get(self, path: str, params: dict = None) -> Any:
        params  = params or {}
        headers = _headers(self.api_key, self.secret_key, query_params=params)
        url     = BASE_URL + path
        resp    = self.session.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def post(self, path: str, body: dict = None) -> Any:
        import json
        body_str = json.dumps(body or {}, separators=(",", ":"))
        headers  = _headers(self.api_key, self.secret_key, body=body_str)
        url      = BASE_URL + path
        resp     = self.session.post(url, headers=headers, data=body_str, timeout=10)
        resp.raise_for_status()
        return resp.json()
