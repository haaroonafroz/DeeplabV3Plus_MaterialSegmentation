from torchvision.datasets import STL10
from typing import Optional, Callable, Union
from pathlib import Path

class SubsetSTL10(STL10):
    def __init__(self, root: Union[str, Path], split: str = 'train', folds: Optional[int] = None,
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
                 download: bool = False, subset_size: Optional[int] = None):
        """
        A subclass of STL10 to provide a subset of the dataset.

        Parameters:
            root (Union[str, Path]): Directory where the dataset will be stored.
            split (str): 'train' for training set, 'test' for test set, 'unlabeled' for unlabeled set.
            folds (Optional[int]): If the split is 'train', 'test' or 'unlabeled', selects which fold to download.
            transform (Optional[Callable]): A function/transform that takes in a PIL image and returns a transformed version.
            target_transform (Optional[Callable]): A function/transform that takes in the target and transforms it.
            download (bool): If True, downloads the dataset from the internet and puts it in root directory.
            subset_size (Optional[int]): If specified, limits the number of samples to subset_size.
        """
        self.root = root
         # Check if the dataset is already downloaded
        if download and not self._check_integrity():
            download = True  # Force download if dataset doesn't exist
            
        super().__init__(root=root, split=split, folds=folds, transform=transform,
                         target_transform=target_transform, download=download)

        # Limit the dataset to only use subset_size number of images
        if subset_size is not None:
            if subset_size < len(self.data):
                self.data = self.data[:subset_size]
                self.labels = self.labels[:subset_size]
            else:
                print(
                    f"Warning: Requested subset size {subset_size} is larger than the dataset size {len(self.data)}. Using full dataset.")