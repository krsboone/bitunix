"""
debug_auth.py — Test auth signature construction against Bitunix

Prints exactly what is being signed so we can compare against
the API docs example if auth is failing.
"""

import hashlib
import json
import random
import string
import time
import requests
from config import API_KEY, SECRET_KEY

BASE_URL = "https://fapi.bitunix.com"


def debug_request():
    nonce     = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    timestamp = str(int(time.time() * 1000))
    params    = {"marginCoin": "USDT"}
    body      = ""

    # Try all three interpretations
    from urllib.parse import urlencode
    param_values_only  = "".join(str(v) for k, v in sorted(params.items()))
    param_urlencode    = urlencode(sorted(params.items()))
    param_keyval_concat = "".join(str(k) + str(v) for k, v in sorted(params.items()))

    print(f"  param (values only)   : '{param_values_only}'")
    print(f"  param (urlencode)     : '{param_urlencode}'")
    print(f"  param (key+val concat): '{param_keyval_concat}'")
    print()

    for label, param_str in [
        ("values only",    param_values_only),
        ("urlencode",      param_urlencode),
        ("key+val concat", param_keyval_concat),
    ]:
        nonce_i     = "".join(random.choices(string.ascii_letters + string.digits, k=8))
        timestamp_i = str(int(time.time() * 1000))
        digest_in   = nonce_i + timestamp_i + API_KEY + param_str + body
        digest_i    = hashlib.sha256(digest_in.encode()).hexdigest()
        sign_i      = hashlib.sha256((digest_i + SECRET_KEY).encode()).hexdigest()

        headers_i = {
            "api-key":      API_KEY,
            "nonce":        nonce_i,
            "timestamp":    timestamp_i,
            "sign":         sign_i,
            "Content-Type": "application/json",
        }
        resp_i = requests.get(
            BASE_URL + "/api/v1/futures/account",
            headers=headers_i,
            params=params,
            timeout=10
        )
        print(f"  [{label}] → HTTP {resp_i.status_code}  {resp_i.text}")

    param_str = param_values_only  # keep original for rest of script (unused)

    digest_input = nonce + timestamp + API_KEY + param_str + body
    digest       = hashlib.sha256(digest_input.encode()).hexdigest()
    sign_input   = digest + SECRET_KEY
    sign         = hashlib.sha256(sign_input.encode()).hexdigest()

    print("── Signature debug ──────────────────────────────────")
    print(f"  api_key   : {API_KEY}")
    print(f"  nonce     : {nonce}")
    print(f"  timestamp : {timestamp}")
    print(f"  params    : {params}")
    print(f"  param_str : '{param_str}'")
    print(f"  body      : '{body}'")
    print(f"  digest_in : '{digest_input[:60]}...'")
    print(f"  digest    : {digest}")
    print(f"  sign      : {sign}")
    print()

    headers = {
        "api-key":      API_KEY,
        "nonce":        nonce,
        "timestamp":    timestamp,
        "sign":         sign,
        "Content-Type": "application/json",
    }

    print("── Request ──────────────────────────────────────────")
    print(f"  GET {BASE_URL}/api/v1/futures/account")
    print(f"  Params: {params}")
    print()

    resp = requests.get(
        BASE_URL + "/api/v1/futures/account",
        headers=headers,
        params=params,
        timeout=10
    )

    print("── Response ─────────────────────────────────────────")
    print(f"  HTTP {resp.status_code}")
    print(f"  Body: {resp.text}")
    print()


if __name__ == "__main__":
    debug_request()
