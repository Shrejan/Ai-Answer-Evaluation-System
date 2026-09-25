import requests

url = "http://10.68.45.34:8001"

with open("imgs/test17.jpg", "rb") as f:
    files = {"file": ("test17.jpg", f, "image/jpg")}
    resp = requests.post(url, files=files)

#import requests

# url = "http://10.68.45.34:8001/ocr"



# with open("imgs/test17.jpg", "rb") as f:
#     files = {"image": ("test17.jpg", f, "image/jpeg")}
#     data = {"page_id": "page_001"}

#     resp = requests.post(url, files=files, data=data)

# print(resp.status_code)
# print(resp.json())

