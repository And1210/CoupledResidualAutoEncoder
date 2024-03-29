import os

import cv2
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from datasets.base_dataset import BaseDataset
from utils.augmenters.augment import seg
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

class CARLADataset(BaseDataset):
    """
    Input params:
        stage: The stage of training.
        configuration: Configuration dictionary.
    """
    def __init__(self, configuration):
        super().__init__(configuration)

        self._stage = configuration["stage"]

        self._image_size = tuple(configuration["input_size"])

        self.dataset_path = os.path.join(configuration["dataset_path"])#, "{}".format(self._stage))

        #-----------------------------------------------------------------------
        #Here is where you can do things like preload data and labels or do image preprocessing

        self.sim_img_paths = []
        self.real_img_paths = []
        for i in os.listdir(os.path.join(self.dataset_path, configuration["sim_data_folder"])):
            self.sim_img_paths.append(os.path.join(self.dataset_path, configuration["sim_data_folder"], i))
        for i in os.listdir(os.path.join(self.dataset_path, configuration["real_data_folder"])):
            self.real_img_paths.append(os.path.join(self.dataset_path, configuration["real_data_folder"], i))

        #-----------------------------------------------------------------------


        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

    #This function returns an data, label pair. All data processing and modification should be done by the end of this function
    def __getitem__(self, index):
        sim_filename = self.sim_img_paths[index]
        real_filename = self.real_img_paths[index]

        #Image loading assuming the images are in the 'images' folder in the dataset root path
        sim_image = Image.open(sim_filename)
        sim_image = np.asarray(sim_image)
        sim_image = sim_image.astype(np.uint8)
        real_image = Image.open(real_filename)
        real_image = np.asarray(real_image)
        real_image = real_image.astype(np.uint8)

        sim_da = match_histograms(sim_image, real_image)
        sim_da = sim_da.astype(np.uint8)

        #Image resizing
        sim_image = cv2.resize(sim_image, self._image_size)
        real_image = cv2.resize(real_image, self._image_size)
        sim_da = cv2.resize(sim_da, self._image_size)

        #Image formatting
        sim_image = np.dstack([sim_image] * 1)
        real_image = np.dstack([real_image] * 1)
        sim_da = np.dstack([sim_da] * 1)

        #Some image augmentation
        # image = seg(image=image)

        #Apply defined transforms to image from constructor (will convert to tensor)
        sim_image = self._transform(sim_image)
        real_image = self._transform(real_image)
        sim_da = self._transform(sim_da)

        #image should be the image data, target should be the label
        return sim_image, real_image, sim_da

    def __len__(self):
        # return the size of the dataset, replace with len of labels array
        return len(self.sim_img_paths)
