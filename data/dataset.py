"""Local dataset loading and image transforms for supported FL experiments."""
from __future__ import annotations
import csv
import gzip
import struct
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)
MNIST_MEAN = (0.1307, 0.1307, 0.1307)
MNIST_STD = (0.3081, 0.3081, 0.3081)
GTSRB_MEAN = (0.3403, 0.3121, 0.3214)
GTSRB_STD = (0.2724, 0.2608, 0.2669)
DATASET_SPECS = {
    "cifar10": {"num_classes": 10, "channels": 3, "image_size": 32, "mean": CIFAR10_MEAN, "std": CIFAR10_STD},
    "cifar100": {"num_classes": 100, "channels": 3, "image_size": 32, "mean": CIFAR100_MEAN, "std": CIFAR100_STD},
    "svhn": {"num_classes": 10, "channels": 3, "image_size": 32, "mean": SVHN_MEAN, "std": SVHN_STD},
    "mnist": {"num_classes": 10, "channels": 3, "image_size": 32, "mean": MNIST_MEAN, "std": MNIST_STD},
    "gtsrb": {"num_classes": 43, "channels": 3, "image_size": 32, "mean": GTSRB_MEAN, "std": GTSRB_STD},
}

def get_dataset_spec(name: str) -> dict:
    """Return normalization and shape metadata for a supported dataset."""
    try:
        return DATASET_SPECS[name.lower()].copy()
    except (AttributeError, KeyError) as error:
        raise ValueError("dataset.name must be 'cifar10', 'cifar100', 'svhn', 'mnist', or 'gtsrb'") from error

def build_cifar10_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Build unchanged CIFAR-10 training and test transforms."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    test_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    return train_transform, test_transform

def _build_32_rgb_transforms(mean: tuple, std: tuple, augment: bool = False):
    steps = [transforms.Resize((32, 32))]
    if augment:
        steps.append(transforms.RandomHorizontalFlip())
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(steps), transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean, std)])

class LocalMNIST(Dataset):
    """MNIST IDX gzip reader using the four files uploaded at datasets root."""
    def __init__(self, root: str | Path, train: bool, transform=None) -> None:
        root = Path(root)
        prefix = "train" if train else "t10k"
        image_path = root / f"{prefix}-images-idx3-ubyte.gz"
        label_path = root / f"{prefix}-labels-idx1-ubyte.gz"
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing local MNIST gzip files: {image_path}, {label_path}")
        with gzip.open(image_path, "rb") as handle:
            magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
            payload = handle.read()
        with gzip.open(label_path, "rb") as handle:
            label_magic, label_count = struct.unpack(">II", handle.read(8))
            labels = handle.read()
        if (magic != 2051 or label_magic != 2049 or count != label_count
                or len(payload) != count * rows * cols or len(labels) != count):
            raise ValueError("Invalid MNIST IDX gzip header or payload length")
        self.data = np.frombuffer(payload, dtype=np.uint8).reshape(count, rows, cols)
        self.targets = [int(value) for value in labels]
        self.transform = transform
    def __len__(self) -> int:
        return len(self.targets)
    def __getitem__(self, index: int):
        image = Image.fromarray(self.data[index], mode="L").convert("RGB")
        return (self.transform(image) if self.transform else image), self.targets[index]

class LocalGTSRB(Dataset):
    """Read GTSRB PPM images from a one-time local extraction cache."""
    TRAIN_ZIP = "GTSRB_Final_Training_Images.zip"
    TEST_ZIP = "GTSRB_Final_Test_Images.zip"
    TEST_GT_ZIP = "GTSRB_Final_Test_GT.zip"
    EXTRACTED_DIR = "gtsrb_extracted"
    EXPECTED_TRAIN_IMAGES = 39209
    EXPECTED_TEST_IMAGES = 12630
    def __init__(self, root: str | Path, split: str, transform=None) -> None:
        self.root, self.split, self.transform = Path(root), split, transform
        self.extracted_root = self._ensure_extracted()
        train_dir = self.extracted_root / "GTSRB/Final_Training/Images"
        test_dir = self.extracted_root / "GTSRB/Final_Test/Images"
        if split == "train":
            self.entries = sorted(train_dir.rglob("*.ppm"))
            self.targets = [int(path.parent.name) for path in self.entries]
        elif split == "test":
            labels = self._read_test_labels(self.extracted_root / "GT-final_test.csv")
            self.entries = sorted(test_dir.glob("*.ppm"))
            self.targets = [labels[path.name] for path in self.entries]
        else:
            raise ValueError("GTSRB split must be 'train' or 'test'")
        expected = self.EXPECTED_TRAIN_IMAGES if split == "train" else self.EXPECTED_TEST_IMAGES
        if len(self.entries) != expected:
            raise ValueError(f"GTSRB {split} extraction has {len(self.entries)} images; expected {expected}")
        if any(label < 0 or label >= 43 for label in self.targets):
            raise ValueError("GTSRB labels must be in [0, 42]")

    def _ensure_extracted(self) -> Path:
        """Extract source archives once, then reuse validated ordinary files."""
        destination = self.root / self.EXTRACTED_DIR
        train_dir = destination / "GTSRB/Final_Training/Images"
        test_dir = destination / "GTSRB/Final_Test/Images"
        test_gt = destination / "GT-final_test.csv"
        complete = (
            train_dir.is_dir() and test_dir.is_dir() and test_gt.is_file()
            and sum(1 for _ in train_dir.rglob("*.ppm")) == self.EXPECTED_TRAIN_IMAGES
            and sum(1 for _ in test_dir.glob("*.ppm")) == self.EXPECTED_TEST_IMAGES
        )
        if complete:
            return destination
        archives = [self.root / self.TRAIN_ZIP, self.root / self.TEST_ZIP, self.root / self.TEST_GT_ZIP]
        missing = [str(path) for path in archives if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing local GTSRB zip files: " + ", ".join(missing))
        destination.mkdir(parents=True, exist_ok=True)
        for archive_path in archives:
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    target = (destination / member.filename).resolve()
                    if destination.resolve() not in (target, *target.parents):
                        raise ValueError(f"Unsafe archive member: {member.filename}")
                archive.extractall(destination)
        return destination
    @staticmethod
    def _read_test_labels(gt_csv: Path) -> dict[str, int]:
        if not gt_csv.is_file():
            raise FileNotFoundError(f"Missing extracted GTSRB labels: {gt_csv}")
        text = gt_csv.read_text(encoding="utf-8-sig")
        return {row["Filename"]: int(row["ClassId"])
                for row in csv.DictReader(text.splitlines(), delimiter=";")}
    def __len__(self) -> int:
        return len(self.entries)
    def __getitem__(self, index: int):
        image = Image.open(self.entries[index]).convert("RGB")
        return (self.transform(image) if self.transform else image), self.targets[index]

def load_cifar10(root: str | Path, download: bool = True) -> tuple[Dataset, Dataset]:
    """Create CIFAR-10 datasets; preserved for backward compatibility."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train_transform, test_transform = build_cifar10_transforms()
    return (datasets.CIFAR10(root=root, train=True, transform=train_transform, download=download),
            datasets.CIFAR10(root=root, train=False, transform=test_transform, download=download))

def load_cifar100(root: str | Path, download: bool = False) -> tuple[Dataset, Dataset]:
    """Create local CIFAR-100 datasets with CIFAR-style augmentation."""
    root = Path(root)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    test_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    return (datasets.CIFAR100(root=root, train=True, transform=train_transform, download=download),
            datasets.CIFAR100(root=root, train=False, transform=test_transform, download=download))

def load_svhn(root: str | Path, download: bool = False) -> tuple[Dataset, Dataset]:
    """Create local SVHN datasets; digits deliberately receive no horizontal flip."""
    root = Path(root)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(SVHN_MEAN, SVHN_STD)])
    train_dataset = datasets.SVHN(root=root, split="train", transform=transform, download=download)
    test_dataset = datasets.SVHN(root=root, split="test", transform=transform, download=download)
    train_dataset.targets = [int(label) for label in train_dataset.labels]
    test_dataset.targets = [int(label) for label in test_dataset.labels]
    return train_dataset, test_dataset

def load_dataset(name: str, root: str | Path, download: bool = False) -> tuple[Dataset, Dataset]:
    """Load a supported dataset locally without forcing network downloads."""
    name = name.lower()
    if name == "cifar10":
        return load_cifar10(root, download=download)
    if name == "cifar100":
        return load_cifar100(root, download=download)
    if name == "svhn":
        return load_svhn(root, download=download)
    spec = get_dataset_spec(name)
    train_transform, test_transform = _build_32_rgb_transforms(
        spec["mean"], spec["std"], augment=name == "gtsrb")
    if name == "mnist":
        return (LocalMNIST(root, train=True, transform=train_transform),
                LocalMNIST(root, train=False, transform=test_transform))
    return (LocalGTSRB(root, split="train", transform=train_transform),
            LocalGTSRB(root, split="test", transform=test_transform))
