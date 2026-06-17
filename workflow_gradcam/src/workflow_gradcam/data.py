from pathlib import Path
import os
import kagglehub
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def download_alzheimer_dataset():
    """Download the Alzheimer dataset through kagglehub.

    Dataset used in the original notebook:
    yasserhessein/dataset-alzheimer
    """
    path = kagglehub.dataset_download("yasserhessein/dataset-alzheimer")
    return Path(path)


def resolve_dataset_path(dataset_path=None):
    if dataset_path is not None:
        dataset_path = Path(dataset_path)
    else:
        base = download_alzheimer_dataset()
        dataset_path = base / "Alzheimer_s Dataset"

    train_dir = dataset_path / "train"
    test_dir = dataset_path / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected train/test folders under: {dataset_path}. "
            "Pass --dataset-path manually if your dataset is stored elsewhere."
        )

    return train_dir, test_dir


def make_generators(train_dir, test_dir, img_size=224, batch_size=16):
    """Create train/test generators without data augmentation.

    This follows the notebook intention: CNN + CAM pipeline, no data augmentation.
    """
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_data = datagen.flow_from_directory(
        str(train_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    test_data = datagen.flow_from_directory(
        str(test_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return train_data, test_data
