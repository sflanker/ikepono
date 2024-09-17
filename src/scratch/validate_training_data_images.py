import PIL

from PIL import Image
import os

training_data_dir = "/mnt/d/scratch_data/mantas/train_valid/kona"
images_verified = []
images_unidentified = []
images_error = []
for root, _, files in os.walk(training_data_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            full_path = os.path.join(root, file)
            try:
                img = Image.open(full_path)
                images_verified.append(full_path)
            except PIL.UnidentifiedImageError:
                print(f"Unidentified image: {full_path}")
                images_unidentified.append(full_path)
            except Exception as e:
                print(f"Error with image: {full_path}, {e}")
                images_error.append(full_path)

print("---------")
print(f"Verified images: {len(images_verified)}")
print(f"Unidentified images: {len(images_unidentified)}")
print(f"Error images: {len(images_error)}")
for img in images_error:
    print(img)
print("---------")
for img in images_unidentified:
    print(img)
print("---------")