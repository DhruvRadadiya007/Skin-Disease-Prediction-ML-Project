import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# -----------------------
# Base archive folder
# -----------------------
# Get current file location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build absolute paths
archive_path = os.path.join(BASE_DIR, "archive")

csv_path = os.path.join(archive_path, "HAM10000_metadata.csv")
img_dir1 = os.path.join(archive_path, "HAM10000_images_part_1")
img_dir2 = os.path.join(archive_path, "HAM10000_images_part_2")

output_train = "data/train"
output_test = "data/test"

# -----------------------
# Load CSV
# -----------------------
df = pd.read_csv(csv_path)

# -----------------------
# Map image paths
# -----------------------
image_paths = {}

for folder in [img_dir1, img_dir2]:
    for img in os.listdir(folder):
        image_id = img.split('.')[0]
        image_paths[image_id] = os.path.join(folder, img)

df['path'] = df['image_id'].map(image_paths.get)

# -----------------------
# Remove missing paths (safety)
# -----------------------
df = df.dropna(subset=['path'])

# -----------------------
# Train-test split
# -----------------------
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['dx'],
    random_state=42
)

# -----------------------
# Create folders
# -----------------------
def create_folders(base_path, classes):
    for c in classes:
        os.makedirs(os.path.join(base_path, c), exist_ok=True)

classes = df['dx'].unique()

create_folders(output_train, classes)
create_folders(output_test, classes)

# -----------------------
# Copy images
# -----------------------
def copy_images(dataframe, output_folder):
    for _, row in dataframe.iterrows():
        src = row['path']
        label = row['dx']

        dst_dir = os.path.join(output_folder, label)
        dst = os.path.join(dst_dir, os.path.basename(src))

        shutil.copy(src, dst)

copy_images(train_df, output_train)
copy_images(test_df, output_test)

print("✅ Data prepared successfully inside /data folder!")