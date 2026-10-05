"""
Smoke test for a running churn API.

Usage:  python3 scripts/smoke_test.py http://localhost:8000

Uses only the standard library, so it runs anywhere Python does. Exits with
code 1 if any check fails, which makes the CI job fail.
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE_URL = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")

HIGH_RISK = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5,
}
LOW_RISK = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 65.0,
}


def parse(raw):
    """Parse a JSON body; error pages (like a 500) are plain text, so fall back to text."""
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode(errors="replace")


def call(method, path, body=None):
    """Return (status_code, parsed_body). Error responses are returned, not raised."""
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        BASE_URL + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, parse(response.read())
    except urllib.error.HTTPError as error:
        return error.code, parse(error.read())


def wait_until_healthy(timeout_seconds=60):
    """The container needs a moment to start, so retry /health for a while."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            if call("GET", "/health")[0] == 200:
                return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)
    return False


def probability(customer):
    status, body = call("POST", "/predict", customer)
    assert status == 200, f"/predict returned {status}: {body}"
    return body["churn_probability"]


failures = []


def check(name, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


print(f"Smoke testing {BASE_URL}")
if not wait_until_healthy():
    print("FAIL  the API never became healthy within 60 seconds")
    sys.exit(1)

status, body = call("GET", "/health")
check("GET /health returns healthy", status == 200 and body == {"status": "healthy"})

try:
    high = probability(HIGH_RISK)
    low = probability(LOW_RISK)
    check("/predict returns valid probabilities", 0 <= high <= 1 and 0 <= low <= 1,
          f"high={high}, low={low}")
    check("high-risk customer scores above low-risk customer", high > low)

    # The container-level version of the serving bug we fixed: changing ONLY
    # the contract must change the prediction.
    two_year = probability({**HIGH_RISK, "Contract": "Two year"})
    check("changing only the contract changes the prediction", two_year != high,
          f"month-to-month={high}, two-year={two_year}")
except AssertionError as error:
    check("/predict works", False, str(error))

status, body = call("POST", "/predict", {**HIGH_RISK, "Contract": "Two years"})
check("a typo in a field is rejected with 422", status == 422)

print(f"\n{'ALL CHECKS PASSED' if not failures else str(len(failures)) + ' CHECK(S) FAILED'}")
sys.exit(1 if failures else 0)