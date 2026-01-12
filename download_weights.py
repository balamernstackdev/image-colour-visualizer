import requests
import os

url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
output_path = "weights/sam_vit_b_01ec64.pth"

if not os.path.exists("weights"):
    os.makedirs("weights")

print(f"Downloading {url}...")
response = requests.get(url, stream=True)
response.raise_for_status()

with open(output_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print(f"Downloaded to {output_path}")
