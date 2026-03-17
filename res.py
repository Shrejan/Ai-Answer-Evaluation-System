import requests

url = "http://localhost:8000/ocr"


with open("test1.jpg", "rb") as f:
    files = {"file": ("test1.jpg", f, "image/jpeg")}
    resp = requests.put(url, files=files)