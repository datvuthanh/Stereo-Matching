import numpy as np
import os
import sys
from scipy.misc import imread


def extract_patch(im, center, size):
    pass


def preprocess(im):
    assert im.ndim == 3
    im -= im.mean(axis=(0, 1))
    im /= im.std(axis=(0, 1))
    return im


def postprocess(x):
    pass


class DataHandler(object):
    def __init__(self, args):
        ''' Handles  preprocessed binary data and KITTI dataset

        Args:
            data_root (str): Path to raw png files
            util_root (str): Path to preprocessed binary data files
            num_tr_img (int): # of training images
            num_val_img (int): # of validation images
            num_val_loc (int): # of val patches
            batch_size (int): Batch size for training
            psz (int): Patch size
            p_size (int): patch size
            half_range: diparity range

        Attributes:
            num_channels (int): Number of image channels (greyscale vs. RGB)
            batch_size (int): batch_size
            psz (int): Patch size
            p_size (int): patch size
            half_range (int): disparity range
            tr_loc (str): location of training data
            val_loc (str): location of validation data
        '''
        self.args = args

        # if kwargs.get("num_val_img") == 0:
        #     self.val_loc = self.tr_loc
        # else:
        #     if verbose:
        #         print("Validating on training set..")

        # if verbose:
        #     print("Training location: {0} \n Validation location: {1}".format(
        #         self.tr_loc[0], self.val_loc[0]))

    def _infer_args(self, filename):
        '''Infer the args based on the pregenerated binaries'''
        fn = filename.split("_")
        fn[-1] = fn[-1].split(".")[0]
        data_split, _, psz, half_range = [fn[0]] + \
            [int(x) for x in fn[1:]]

        num_channels = 3 if self.args["data_version"] == "kitti2015" else 1
        image_shape = (375, 1242, ) + (num_channels, )
        l_psz = (psz*2+1, psz*2+1, num_channels)
        r_psz = (psz*2+1, psz*2+1+half_range*2, num_channels)

        d = {"psz": psz,
             "p_size": psz*2+1,
             "data_split": data_split,
             "half_range": half_range,
             "num_channels": num_channels,
             "image_shape": image_shape,
             "l_psz": l_psz,
             "r_psz": r_psz}

        self.args.update(d)

    def load(self, filename=None, random_shuffle=True, bin_filename="myPerm.bin"):
        '''
        The hardcoded binary files produced by the matlab scripts are:
            myPerm.bin: Contains the filename permutations
            tr_160_18_100.bin: Contains all pixel locations for all images
            in the left image and the corresponding ground truth for training
            val_40_18_100.bin: Contains all pixel locations for all images
            in the left image and the corresponding ground truth for validation
        '''
        bin_path = os.path.join(self.args["util_root"], bin_filename)

        # Read binarys
        print("Reading binarys..")
        self.file_ids = np.fromfile(bin_path, '<f4').astype(int)
        if filename:
            data_path = os.path.join(self.args["util_root"], filename)
            self.pixel_loc = np.fromfile(
                data_path, '<f4').reshape(-1, 5).astype(np.int)
        # Correct for index assigment in Matlab
        if filename:
            self._correct_index(self.pixel_loc)
        # Shuffle in place
        if random_shuffle and filename:
            np.random.shuffle(self.pixel_loc)

        # Read images
        self.ldata, self.rdata = {}, {}
        print("Reading images..")
        # TODO : Refactor this to only load val or train images for specific
        # split
        num_images = self.args["num_tr_img"] + self.args["num_val_img"]
        for idx in range(num_images):
            fn = self.file_ids[idx]
            try:
                self.ldata[fn], self.rdata[fn] = self._load_sample(fn)
            except Exception as e:
                print("Loading error: {}".format(fn))
                raise e

        # Initialize params
        if filename:
            self._infer_args(filename)
            self._initialize_batch_arrays()
            if self.args["data_split"] == "val" and filename:
                self.pixel_loc = self.pixel_loc[:self.args["num_val_loc"]]

    @property
    def generator(self):
        self._create_label_arrays()
        while True:
            sample_counter = 0
            sample_max = self.args["batch_size"]
            for i in range(self.pixel_loc.shape[0]):
                sample_specs = tuple(self.pixel_loc[i])
                img_id = sample_specs[0]

                # img_left, img_right
                i_l, i_r = self.ldata[img_id], self.rdata[img_id]

                try:
                    self.batch_left[sample_counter] = self._extract_patch(
                        i_l, sample_specs, 0, "left")
                    self.batch_right[sample_counter] = self._extract_patch(
                        i_r, sample_specs, self.args["half_range"], "right")
                except Exception as e:
                    print("Error in extracting patches..")
                    raise e

                sample_counter += 1

                if sample_counter == sample_max:
                    yield [self.batch_left, self.batch_right], self.batch_labels
                    # Reset
                    sample_counter = 0
                    self._initialize_batch_arrays()

    def _extract_patch(self, x, sample_specs, half_range, side):
        img_id, loc_type, center_x, center_y, right_center_x = sample_specs
        right_center_y = center_y

        if side == "right":
            center_y, center_x = right_center_y, right_center_x

        # Loc type always 1, so use the horizontal
        p = x[center_y-self.args["psz"]:
              center_y+self.args["psz"]+1,
              center_x-self.args["psz"]-half_range:
              center_x+self.args["psz"]+half_range+1]

        return p

    def _load_sample(self, fn):
        l_path = os.path.join(
            self.args["data_root"],
            "image_2",
            "%06d_10.png" % fn)
        r_path = os.path.join(
            self.args["data_root"],
            "image_3",
            "%06d_10.png" % fn)

        l_img, r_img = imread(l_path), imread(r_path)
        l_img = preprocess(l_img.astype(np.float32))
        r_img = preprocess(r_img.astype(np.float32))

        return l_img, r_img

    def _initialize_batch_arrays(self):
        # Base shape
        b_s = (self.args["batch_size"],
               self.args["num_channels"], self.args["p_size"], )

        self.batch_left = np.zeros(
            b_s + (self.args["p_size"],),
            np.float32).transpose(0, 2, 3, 1)

        self.batch_right = np.zeros(
            b_s + (self.args["p_size"]+self.args["half_range"]*2,),
            np.float32).transpose(0, 2, 3, 1)

    def _create_label_arrays(self, lambdas=(0.5, 0.2, 0.05)):
        # We add 1 in order to get a well-defined center
        self.batch_labels = np.zeros(
            (self.args["batch_size"], self.args["half_range"]*2 + 1),
            np.float32)

        idx = (self.args["half_range"] - (len(lambdas)-1),
               self.args["half_range"] + (len(lambdas)))

        # Add the smoothed loss around the ground truth
        self.batch_labels[:, idx[0]:idx[1]] = lambdas[::-1] + lambdas[1:]

        # Test if most likely is in center of array
        np.testing.assert_equal(
            np.where(self.batch_labels == lambdas[0])[1],
            np.array([self.args["half_range"]]*self.args["batch_size"]))

    @staticmethod
    def _correct_index(x):
        assert x.ndim == 2 and x.shape[1] == 5
        x[:, 2:5] = x[:, 2:5] - 1


if __name__ == "__main__":
    sys.path.append("../viz_utils")
    import visualization as viz
    import IPython

    data_lookup = {
        "train": "tr_160_18_100.bin",
        "val": "val_40_18_100.bin"
    }

    args = {
        "batch_size": 8,
        "data_version": "kitti2015",
        "util_root": "/home/marco/repos/EfficientStereoMatching/data/KITTI2015/debug_15/",
        "data_root": "/home/marco/repos/EfficientStereoMatching/data/KITTI2015/data_scene_flow/training",
        "filename": data_lookup["train"]
    }

    dh = DataHandler(args)
    dh.load()
    gen = dh.generator
    IPython.embed()
    x, y, labels = gen.next()
    viz.plot_pairwise_training_batches(x[0:10], y[0:10])
    print(labels[0])
    viz.plt.show()
