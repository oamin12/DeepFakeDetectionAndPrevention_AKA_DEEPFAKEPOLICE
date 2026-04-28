from pathlib import Path
import shutil
import pandas as pd

BASE_DIR = Path("Datasets/nuriachandra_Deepfake-Eval-2024")

csv_path = BASE_DIR / "image-metadata-publish.csv"
images_dir = BASE_DIR / "image-data"

real_dir = BASE_DIR / "real"
fake_dir = BASE_DIR / "fake"

real_dir.mkdir(exist_ok=True)
fake_dir.mkdir(exist_ok=True)

df = pd.read_csv(csv_path)

for _, row in df.iterrows():
    filename = row["Filename"]
    label = str(row["Ground Truth"]).strip().lower()

    src = images_dir / filename

    if label == "real":
        dst = real_dir / filename
    elif label == "fake":
        dst = fake_dir / filename
    else:
        print(f"Skipping unknown label: {label} for {filename}")
        continue

    if src.exists():
        shutil.copy2(src, dst)
    else:
        print(f"Missing file: {src}")

print("Done splitting images into real/ and fake/")