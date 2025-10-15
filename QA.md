# QA Guide

## Bubble Data API Smoke Test

To verify connectivity with Bubble's Data API and the helper functions, run the
following script from the repository root (after ensuring `requests` is
installed and outbound network access is permitted):

```bash
BUBBLE_API_KEY="<your-api-key>" \
BUBBLE_API_BASE="https://system.vanlee.co.jp/version-test/api/1.1" \
python - <<'PY'
from app import bubble_client

response = bubble_client.bubble_search(
    "Receipt",
    limit=1,
)
print(response)
PY
```

Alternatively, you can use `curl`:

```bash
curl \
  -H "Authorization: Bearer ${BUBBLE_API_KEY}" \
  -H "Content-Type: application/json" \
  "${BUBBLE_API_BASE:-https://system.vanlee.co.jp/version-test/api/1.1}/obj/Receipt?limit=1"
```

A successful test should return a 2xx status code and include a JSON payload
from Bubble containing up to one `Receipt` record.
