import numpy as np

from advent.dataset.base_dataset import BaseDataset
from advent.utils.serialization import json_load
from advent.utils import project_root

DEFAULT_INFO_PATH = project_root / 'advent/dataset/Vaihingen/info.json'

class Potsdam(BaseDataset):
    def __init__(self, root, list_path, set='all',
                 max_iters=None, crop_size=(512, 512), mean=(128, 128, 128),
                 info_path = DEFAULT_INFO_PATH):
        super().__init__(root, list_path, set, max_iters, crop_size, None, mean)

        self.info = json_load(info_path)
        self.class_names = np.array(self.info['label'], dtype=np.str)

    def get_metadata(self, name):
        img_file = self.root / 'images' / name
        label_file = self.root / 'labels' / name
        return img_file, label_file

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        image = self.get_image(img_file)
        label = self.get_labels(label_file)
        image = self.preprocess(image)
        return image.copy(), label-1, np.array(image.shape), name
