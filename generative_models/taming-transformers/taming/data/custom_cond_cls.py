import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths_labels = f.read().splitlines()
        paths, class_labels = zip(*[line.split(",") for line in paths_labels])
        # paths = [os.path.abspath(path) for path in paths]
        class_labels = [int(label) for label in class_labels]
        labels = {
            "class_label": np.array(class_labels),
        }
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, labels=labels)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths_labels = f.read().splitlines()
        paths, class_labels = zip(*[line.split(",") for line in paths_labels])
        # paths = [os.path.abspath(path) for path in paths]
        class_labels = [int(label) for label in class_labels]
        labels = {
            "class_label": np.array(class_labels),
        }
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, labels=labels)


