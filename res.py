import requests

url = "http://localhost:8000/ocr"

with open("imgs/test8.jpg", "rb") as f:
    files = {"file": ("test8.jpg", f, "image/jpeg")}
    resp = requests.post(url, files=files)

'''resp1 = requests.get("http://localhost:8000/health")
print(resp1.json())

resp2 = requests.get("http://localhost:8000/config")
print(resp2.json())'''