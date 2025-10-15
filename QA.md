# QA Guide

## Bubble Data API Smoke Test

To verify connectivity with Bubble's Data API and the new helper functions, you can run:

```bash
python - <<'PY'
import requests

response = requests.get(
    "https://system.vanlee.co.jp/version-test/api/1.1/obj/Receipt",
    headers={
        "Authorization": "Bearer 99c107353970e7d1f2f1b36709cd3e04",
        "Content-Type": "application/json",
    },
    params={"limit": 1},
    timeout=20,
)
print(response.status_code)
response.raise_for_status()
print(response.json())
PY
```

A successful test should print a 2xx status code followed by the JSON response body.
