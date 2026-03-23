import requests

url = "http://localhost:8000/ocr"

with open("imgs/test8.jpg", "rb") as f:
    files = {"file": ("test8.jpg", f, "image/jpeg")}
    resp = requests.post(url, files=files)

print(resp.status_code)
