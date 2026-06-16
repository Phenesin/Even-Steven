from pathlib import Path
from urllib.request import urlretrieve
from config import DATA_DIR, CLASSES


BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"


output_dir = Path(DATA_DIR)
output_dir.mkdir(parents = True, exist_ok = True)

for class_name in CLASSES:
    destination = output_dir / f"{class_name}.npy"
    if destination.exists():
        print(f"[SKIP]{class_name}")
        continue
    url = f"{BASE_URL}/{class_name}.npy"
    print(url)
    print(f"[DOWNLOAD]{class_name}")
    urlretrieve(url, destination)
print("Done.")