import shutil
from pathlib import Path
from sklearn.model_selection import KFold

# Path configuration
BASE_DIR = Path(".")  # Current directory
IMAGE_DIR = BASE_DIR / "dataset/BaseImages"

# List of your 3 different annotation variant folders
ANNOTATION_DIRS = {
    "datasetManual": BASE_DIR / "dataset/CleanManualAnnotations",
    "datasetGrabcut": BASE_DIR / "dataset/CleanGrabcutAnnotations",
    "datasetSAM2": BASE_DIR / "dataset/CleanSAM2Annotations"
}

OUTPUT_DIR = BASE_DIR / "crossValidationFolder"
NUM_SPLITS = 5
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}

# Get all valid images and sort them so the order is deterministic
all_images = sorted([f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])

# Initialize KFold with a fixed random state so it's perfectly reproducible
kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=451)

# Splitting and copying
for fold_idx, (train_indices, val_indices) in enumerate(kf.split(all_images), 1):
    print(f"\nProcessing Fold {fold_idx}/{NUM_SPLITS}...")

    # Define splits for this fold
    train_images = [all_images[i] for i in train_indices]
    val_images = [all_images[i] for i in val_indices]

    # Process each annotation variant using this fold's exact split
    for dataset_name, ann_dir in ANNOTATION_DIRS.items():

        # Construct directory paths: e.g., crossValidationFolder/Fold_1/dataset_v1/images/train
        fold_train_img_dir = OUTPUT_DIR / f"Fold_{fold_idx}" / dataset_name / "images" / "train"
        fold_train_msk_dir = OUTPUT_DIR / f"Fold_{fold_idx}" / dataset_name / "masks" / "train"
        fold_val_img_dir = OUTPUT_DIR / f"Fold_{fold_idx}" / dataset_name / "images" / "val"
        fold_val_msk_dir = OUTPUT_DIR / f"Fold_{fold_idx}" / dataset_name / "masks" / "val"

        # Create directories safely
        for d in [fold_train_img_dir, fold_train_msk_dir, fold_val_img_dir, fold_val_msk_dir]:
            d.mkdir(parents=True, exist_ok=True)


        # Helper function to copy images and their respective masks
        def copy_split_data(image_list, target_img_dir, target_msk_dir):
            for img_path in image_list:
                # Copy the raw image
                shutil.copy2(img_path, target_img_dir / img_path.name)

                # Look for the corresponding mask (assuming same file name)
                mask_name = img_path.stem + ".png"
                mask_path = ann_dir / mask_name

                if mask_path.exists():
                    shutil.copy2(mask_path, target_msk_dir / mask_name)
                else:
                    print(f"Missing mask for {img_path.name} in {dataset_name}")


        # Copy the files over
        copy_split_data(train_images, fold_train_img_dir, fold_train_msk_dir)
        copy_split_data(val_images, fold_val_img_dir, fold_val_msk_dir)

print(f"Successfully generated 5 identical folds inside: {OUTPUT_DIR}")