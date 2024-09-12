import numpy as np
import os
from pathlib import Path


class BuildTrainValidateDirectories:

    @staticmethod
    def build(images_dir : Path, train_dir : Path, valid_dir : Path, k: int, test_percent : float):
        # Create the train and valid directories
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)

        # Get the list of directories in images_dir
        source_dirs = [d for d in images_dir.iterdir() if d.is_dir()]

        for s in source_dirs:
            BuildTrainValidateDirectories.build_train_valid(s, train_dir, valid_dir, k, test_percent)

    @staticmethod
    def build_train_valid(source_dir, train_dir, valid_dir, k, test_percent):
        # Get the list of images in the source directory
        images = [i for i in source_dir.iterdir() if i.is_file()]
        if len(images) < k:
            return
        # Shuffle image (in place)
        np.random.shuffle(images)
        # Always put at least one image in the validation set
        num_test = max(1, int(len(images) * test_percent))
        num_train = len(images) - num_test

        test_images = images[:num_test]
        train_images = images[num_test:]

        # Make new directories for the source
        train_source_dir = train_dir / source_dir.name
        valid_source_dir = valid_dir / source_dir.name
        os.makedirs(train_source_dir, exist_ok=True)
        os.makedirs(valid_source_dir, exist_ok=True)

        for images in train_images:
            os.symlink(images, train_source_dir / images.name)
        for images in test_images:
            os.symlink(images, valid_source_dir / images.name)

if __name__ == "__main__":
    images_dir = Path("/mnt/d/scratch_data/mantas/by_name/inner_crop/kona")
    output_dir = Path("/mnt/d/scratch_data/mantas/train_valid/inner_crop/kona")

    train_dir = output_dir / "train"
    valid_dir = output_dir / "valid"

    BuildTrainValidateDirectories.build(images_dir, train_dir, valid_dir, 5, 0.2)