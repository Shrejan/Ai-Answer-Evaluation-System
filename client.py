"""
Simple client for the OCR Reconstruction API.

Sends a noisy OCR text payload to the /reconstruct endpoint and prints
raw, preprocessed, and reconstructed output.
"""

from __future__ import annotations

import sys

import requests

# ---------------------------------------------------------------------------
# Configuration – set your noisy OCR text here
# ---------------------------------------------------------------------------

OCR_TEXT = '''shashworthies .
half-school .
write the advantages of this and what are helpful
advantages of them .
" High data " transfer speed
show communication cost
" Easy charming of files and printers
Every network management
" better faculty .
" Early Communication between users
" Resource chairing in possible
types of land .
swoked Lane .
In what they compares one connected voting
cably such as the most cable . It provides high
speed and stable communication
quickless land .
an this duties are launched using cattle ,
whiff without cables to " proudly mobility and
flexibility .
relent serum land .
in this one central former controls increase
returns and moulds motistry , and fluttility sources'''

API_URL = "http://localhost:8000/reconstruct"
HEALTH_URL = "http://localhost:8000/health"
TIMEOUT_SECONDS = 120


def check_health() -> bool:
    """Verify the API server is running and ready."""
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"Server status : {data.get('status')}")
        print(f"Model         : {data.get('model')}")
        print(f"Device        : {data.get('device')}")
        print(f"GPU available : {data.get('gpu_available')}")
        print()
        return data.get("status") == "ok"
    except requests.ConnectionError:
        print("ERROR: Cannot connect to the API server.")
        print("Start the server first:  python main.py")
        return False
    except requests.RequestException as exc:
        print(f"ERROR: Health check failed: {exc}")
        return False


def reconstruct(text: str) -> None:
    """Send OCR text to the reconstruction endpoint and display results."""
    payload = {"text": text}

    print("=" * 60)
    print("OCR RECONSTRUCTION CLIENT")
    print("=" * 60)
    print()

    if not check_health():
        sys.exit(1)

    print(f"Sending text ({len(text)} chars)...")
    print(f'Input: "{text}"')
    print()

    try:
        response = requests.post(API_URL, json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.ConnectionError:
        print("ERROR: Lost connection to the API server.")
        sys.exit(1)
    except requests.Timeout:
        print("ERROR: Request timed out. The model may still be loading.")
        sys.exit(1)
    except requests.HTTPError as exc:
        print(f"ERROR: API returned {exc.response.status_code}: {exc.response.text}")
        sys.exit(1)

    data = response.json()

    print("-" * 60)
    print("RAW TEXT")
    print("-" * 60)
    print(data["raw"])
    print()

    print("-" * 60)
    print("PREPROCESSED TEXT")
    print("-" * 60)
    print(data["preprocessed"])
    print()

    print("-" * 60)
    print("RECONSTRUCTED TEXT")
    print("-" * 60)
    print(data["reconstructed"])
    print()

    print("-" * 60)
    print("METADATA")
    print("-" * 60)
    print(f"Confidence         : {data.get('confidence', 'N/A')}")
    print(f"Processing time    : {data.get('processing_time_ms', 'N/A')} ms")
    print("=" * 60)


if __name__ == "__main__":
    reconstruct(OCR_TEXT)
