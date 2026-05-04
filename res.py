import requests

url = "http://localhost:8000/ocr"

with open("imgs/test11.jpeg", "rb") as f:
    files = {"file": ("test11.jpeg", f, "image/jpeg")}
    resp = requests.post(url, files=files)


