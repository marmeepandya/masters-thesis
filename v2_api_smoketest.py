#!/usr/bin/env python
"""Minimal, quota-safe smoke test for the Istari v2 GOI API.

Checks (in increasing cost order):
  1. GET  /health          - liveness, no quota cost
  2. POST /search  size=3  - tiny search, reports rate-limit headers so we
                             can see actual remaining quota after the call

Run manually: python v2_api_smoketest.py
"""
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = 'https://api.istari.ai/v2'
API_KEY = os.getenv('API_KEY')

if not API_KEY:
    raise SystemExit('API_KEY not set in .env')

HEADERS = {
    'Accept': 'application/json',
    'x-api-key': API_KEY,
    'Content-Type': 'application/json',
}


def check_health():
    print('== GET /health ==')
    r = requests.get(f'{BASE_URL}/health', headers=HEADERS, timeout=10)
    print(f'status: {r.status_code}')
    try:
        print(json.dumps(r.json(), indent=2))
    except ValueError:
        print(r.text[:500])
    print()


def tiny_search():
    print('== POST /search (size=3, describe="software companies") ==')
    body = {
        'describe': 'software companies',
        'keywords': {'must_all': [], 'must_any': [], 'must_not': []},
        'filters': {
            'country': [], 'state': [], 'region': [],
            'organization_type': [], 'organization_size': [], 'nace_code': [],
        },
        'excludes': [],
        'columns': ['domain', 'name', 'country'],
        'size': 3,
    }
    r = requests.post(f'{BASE_URL}/search', headers=HEADERS, json=body, timeout=15)
    print(f'status: {r.status_code}')

    rl_keys = [
        'X-RateLimit-Requests-Limit', 'X-RateLimit-Requests-Remaining',
        'X-RateLimit-Results-Limit', 'X-RateLimit-Results-Remaining',
        'X-RateLimit-Reset',
    ]
    rl = {k: r.headers[k] for k in rl_keys if k in r.headers}
    if rl:
        print('rate limit headers:')
        for k, v in rl.items():
            print(f'  {k}: {v}')
    else:
        print('(no rate-limit headers in response)')

    try:
        payload = r.json()
    except ValueError:
        print(r.text[:1000])
        return
    print(f'\nmode: {payload.get("metadata", {}).get("mode")}')
    print(f'total_hits: {payload.get("metadata", {}).get("total_hits")}')
    print(f'elapsed_ms: {payload.get("metadata", {}).get("elapsed_ms")}')
    print('data:')
    print(json.dumps(payload.get('data', []), indent=2))


if __name__ == '__main__':
    check_health()
    tiny_search()
