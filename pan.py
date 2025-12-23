import os
import shutil
import random

# CONFIG
SOURCE_PAN = r"C:\Users\Asus\Downloads\PAN"
TARGET_PAN = r"data\processed\pan"

def fix_pan_dataset():
    # 1. Create clean structure
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(TARGET_PAN, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(TARGET_PAN, 'labels', split), exist_ok=True)

    # 2. Copy all Train files
    for folder in ['images', 'labels']:
        src = os.path.join(SOURCE_PAN, 'train', folder)
        dst = os.path.join(TARGET_PAN, folder, 'train')
        for f in os.listdir(src):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

    # 3. Handle Valid -> Test Split
    # We take 100 images from the 268 valid images for Testing
    val_imgs_path = os.path.join(SOURCE_PAN, 'valid', 'images')
    all_val_files = os.listdir(val_imgs_path)
    test_samples = random.sample(all_val_files, 100)
    
    for f in all_val_files:
        split = 'test' if f in test_samples else 'valid'
        
        # Copy Image
        shutil.copy2(os.path.join(val_imgs_path, f), 
                     os.path.join(TARGET_PAN, 'images', split, f))
        
        # Copy Label
        lbl_name = f.rsplit('.', 1)[0] + ".txt"
        shutil.copy2(os.path.join(SOURCE_PAN, 'valid', 'labels', lbl_name), 
                     os.path.join(TARGET_PAN, 'labels', split, lbl_name))

    # 4. Create the Professional data.yaml
    yaml_content = f"""
path: {os.path.abspath(TARGET_PAN)}
train: images/train
val: images/valid
test: images/test

nc: 4
names:
  0: dob
  1: father_name
  2: name
  3: pan_number
"""
    with open(os.path.join(TARGET_PAN, 'data.yaml'), 'w') as f:
        f.write(yaml_content.strip())

    print(f"Done! PAN dataset ready.")
    print(f"Train: 1839 | Valid: 168 | Test: 100")

if __name__ == "__main__":
    fix_pan_dataset()