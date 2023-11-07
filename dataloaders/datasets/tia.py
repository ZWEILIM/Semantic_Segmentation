from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
# import custom_transforms as tr
import subprocess
import sys
import random




class Segmentation(Dataset):
    NUM_CLASSES = 1

    def __init__(self,
                 args,
                 split='train',  # Add 'split' as an argument with a default value
                 base_dir=Path.db_root_dir('7b52009b64fd0a2a49e6d8a939753077792b0554')):
        """
        :param base_dir: path to dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = Path.db_root_dir(args.dataset)
        self._image_dir = os.path.join(self._base_dir, 'test','images')
        self._cat_dir = os.path.join(self._base_dir, 'test','gt')
        # self.transform = transform  # Store the transform as an instance variable

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args


        # self.images = []  # List to store image file paths
        self.im_ids = []
        self.images = []
        self.categories = []

        for filename in os.listdir(self._image_dir):
            if filename.endswith("_sat.png"):  # Adjust the file extension accordingly
                frame_number_str = filename.split("_sat.png")[0]  # Extract frame number without "_sat.png" suffix
                img_path = os.path.join(self._image_dir, filename)
                self.images.append(img_path)
                self.im_ids.append(frame_number_str)  # Store the extracted frame number


        self.categories = []  # List to store category (mask) file paths
        for filename in os.listdir(self._cat_dir):
            if filename.endswith("_gt.png"):  # Adjust the file extension accordingly
                cat_path = os.path.join(self._cat_dir, filename)
                self.categories.append(cat_path)

        assert len(self.images) == len(self.categories)

        random.shuffle(self.images)
        random.shuffle(self.categories)

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))
        print("Checking for None values in self.im_ids:", any(item is None for item in self.im_ids))
        print("Checking for None values in self.images:", any(item is None for item in self.images))
        print("Checking for None values in self.categories:", any(item is None for item in self.categories))
        # print("Checking for None values in self.connect_label:", any(item is None for item in self.connect_label))
        # print("Checking for None values in self.connect_d1_label:", any(item is None for item in self.connect_d1_label))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        # print("Sample:", sample)  # Print the sample before returning


        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample), self.im_ids[index]
            elif split == 'test':
                return self.transform_test(sample), self.im_ids[index]


    def _make_img_gt_point_pair(self, index):
        # frame_number = index  # Assuming frame numbers are the same as the index
        frame_number_str = self.im_ids[index]  # Extract the frame number from the im_ids list

        img_path = os.path.join(self._image_dir, f"{frame_number_str}_sat.png")
        target_path = os.path.join(self._cat_dir, f"{frame_number_str}_gt.png")
        # connect0_path = os.path.join(self._con_dir, f"{frame_number_str}_gt_0.png")
        # connect1_path = os.path.join(self._con_dir, f"{frame_number_str}_gt_1.png")
        # connect2_path = os.path.join(self._con_dir, f"{frame_number_str}_gt_2.png")
        # connect_d1_0_path = os.path.join(self._con_d1_dir, f"{frame_number_str}_gt_0.png")
        # connect_d1_1_path = os.path.join(self._con_d1_dir, f"{frame_number_str}_gt_1.png")
        # connect_d1_2_path = os.path.join(self._con_d1_dir, f"{frame_number_str}_gt_2.png")

        # print("Image path:", img_path)
        # print("Target path:", target_path)
        # print("Connect0 path:", connect0_path)
        # print("connect_d1_0 path:", connect_d1_0_path)
        # ... print other paths ...

        _img = Image.open(img_path).convert('RGB')
        _target = Image.open(target_path)
        # _connect0 = Image.open(connect0_path).convert('RGB')
        # _connect1 = Image.open(connect1_path).convert('RGB')
        # _connect2 = Image.open(connect2_path).convert('RGB')
        # _connect_d1_0 = Image.open(connect_d1_0_path).convert('RGB')
        # _connect_d1_1 = Image.open(connect_d1_1_path).convert('RGB')
        # _connect_d1_2 = Image.open(connect_d1_2_path).convert('RGB')

        return _img, _target #, _connect0, _connect1, _connect2, _connect_d1_0, _connect_d1_1, _connect_d1_2

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomRotate(180),
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_test(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize_test(size=self.args.crop_size),
            tr.Normalize_test(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor_test()
        ])

        return composed_transforms(sample)


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # args.base_size = 512
    # args.crop_size = 512
    args.base_size = 280
    args.crop_size = 280
    args.batch_size = 1
    args.dataset = '7b52009b64fd0a2a49e6d8a939753077792b0554'

    data_train = Segmentation(args, split='test')

    dataloader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='7b52009b64fd0a2a49e6d8a939753077792b0554')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


