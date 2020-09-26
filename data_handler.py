import os

import numpy as np
from scipy.misc import imread


class DataHandler(object):
    def __init__(self, batch_size, data_version, util_root, data_root, filename):
        self.batch_size = batch_size
        self.data_version = data_version
        self.util_root = util_root
        self.data_root = data_root
        self.filename = filename
        self.num_channels = 3 if self.data_version == "kitti2015" else 1

        fn = filename.split("_")
        fn[-1] = fn[-1].split(".")[0]
        self.num_tr_img = int(fn[1])
        self.num_val_img = int(fn[2])
        self.num_val_loc = int(fn[3])
        data_split, _, psz, half_range = [fn[0]] + [int(x) for x in fn[1:]]
        self.data_split = data_split
        self.half_range = half_range
        self.psz = psz
        self.p_size = psz*2+1
        self.l_psz = (psz*2+1, psz*2+1, self.num_channels)
        self.r_psz = (psz*2+1, psz*2+1+half_range*2, self.num_channels)
        self.image_shape = (375, 1242, ) + (self.num_channels, )


    def load(self, shuffle=True, bin_filename="myPerm.bin"):
        bin_path = os.path.join(self.util_root, bin_filename)

        print("Parse %s" % bin_filename)
        self.file_ids = np.fromfile(bin_path, '<f4').astype(int)
        data_path = os.path.join(self.util_root, self.filename)
        self.pixel_loc = np.fromfile(data_path, '<f4').reshape(-1, 5).astype(np.int)
        self.pixel_loc[:, 2:5] = self.pixel_loc[:, 2:5] - 1
        if shuffle:
            np.random.shuffle(self.pixel_loc)

        self.ldata, self.rdata = {}, {}
        print("Loading images")
        num_images = self.num_tr_img + self.num_val_img
        for idx in range(num_images):
            fn = self.file_ids[idx]
            self.ldata[fn], self.rdata[fn] = self.load_sample(fn)

        self.init_input_batches()
        if self.data_split == "val":
            self.pixel_loc = self.pixel_loc[:self.num_val_loc]

    def preprocess_image(self, im):
        im -= im.mean(axis=(0, 1))
        im /= im.std(axis=(0, 1))
        return im

    @property
    def generator(self):
        self.init_labels()
        while True:
            sample_counter = 0
            sample_max = self.batch_size
            for i in range(self.pixel_loc.shape[0]):
                sample_specs = tuple(self.pixel_loc[i])
                img_id = sample_specs[0]
                img_left, img_right = self.ldata[img_id], self.rdata[img_id]

                self.batch_left[sample_counter] = self.extract_patch(img_left, sample_specs, 0, "left")
                self.batch_right[sample_counter] = self.extract_patch(img_right, sample_specs, self.half_range, "right")

                sample_counter += 1
                if sample_counter == sample_max:
                    yield [self.batch_left, self.batch_right], self.batch_labels
                    sample_counter = 0
                    self.init_input_batches()

    def extract_patch(self, x, sample_specs, half_range, side):
        img_id, loc_type, center_x, center_y, right_center_x = sample_specs
        right_center_y = center_y

        if side == "right":
            center_y, center_x = right_center_y, right_center_x

        patch = x[center_y-self.psz: center_y+self.psz+1, center_x-self.psz-half_range: center_x+self.psz+half_range+1]
        return patch

    def load_sample(self, fn):
        l_path = os.path.join(
            self.data_root,
            "image_2",
            "%06d_10.png" % fn)
        r_path = os.path.join(
            self.data_root,
            "image_3",
            "%06d_10.png" % fn)

        l_img, r_img = imread(l_path), imread(r_path)
        l_img = self.preprocess_image(l_img.astype(np.float32))
        r_img = self.preprocess_image(r_img.astype(np.float32))

        return l_img, r_img

    def init_input_batches(self):
        base_shape = (self.batch_size, self.num_channels, self.p_size)
        self.batch_left = np.zeros(base_shape + (self.p_size,), np.float32).transpose(0, 2, 3, 1)
        self.batch_right = np.zeros(base_shape + (self.p_size+self.half_range*2,), np.float32).transpose(0, 2, 3, 1)

    def init_labels(self, lambdas=(0.5, 0.2, 0.05)):
        self.batch_labels = np.zeros((self.batch_size, self.half_range*2 + 1), np.float32)
        idx = (self.half_range - (len(lambdas)-1), self.half_range + (len(lambdas)))

        # Add the smoothed loss around the ground truth
        self.batch_labels[:, idx[0]:idx[1]] = lambdas[::-1] + lambdas[1:]
