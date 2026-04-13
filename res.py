import requests

url = "http://localhost:8000/ocr"

with open("imgs/test2.jpg", "rb") as f:
    files = {"file": ("test2.jpg", f, "image/jpeg")}
    resp = requests.post(url, files=files)

print(resp.status_code)

