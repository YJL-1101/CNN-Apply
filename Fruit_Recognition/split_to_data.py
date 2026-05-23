import os
import random
import shutil

SRC = './fruit11'
DST = './data'
TMP = DST + "_tmp"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

classes = sorted(os.listdir(SRC))

for cls in classes:
    src_dir = os.path.join(SRC, cls)
    images = sorted(os.listdir(src_dir))
    random.shuffle(images)

    total = len(images)
    n_train = int(total * TRAIN_RATIO)

    train_dst = os.path.join(TMP, "Train", cls)
    test_dst = os.path.join(TMP, "Test", cls)
    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(test_dst, exist_ok=True)

    for i, img in enumerate(images):
        src_path = os.path.join(src_dir, img)
        if i < n_train:
            dst_path = os.path.join(train_dst, img)
        else:
            dst_path = os.path.join(test_dst, img)
        shutil.copy(src_path, dst_path)

    print(f"  {cls:15s}  Train={n_train:4d}  Test={total - n_train:4d}  Total={total}")

if os.path.isdir(DST):
    bak = DST + "_bak"
    if os.path.isdir(bak):
        shutil.rmtree(bak, ignore_errors=True)
    os.rename(DST, bak)

os.rename(TMP, DST)

if os.path.isdir(DST + "_bak"):
    shutil.rmtree(DST + "_bak", ignore_errors=True)

print(f"\nDONE!  data/Train/<class>/  +  data/Test/<class>/")
print("fruit11 unchanged.")
