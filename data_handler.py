import numpy as np
import sys
import os
from scipy import misc


class Data_handler:

    def __init__(self, data_version, data_root, util_root, num_tr_img, \
        num_val_img, num_val_loc, batch_size, patch_size, disp_range):

        if data_version == 'kitti2015':
            self.num_channels = 3
        elif data_version == 'kitti2012':
            self.num_channels = 1
        else:
            sys.exit('data_version should be either kitti2012 or kitti2015')
        
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.disp_range = disp_range
        self.half_patch = patch_size // 2
        self.half_range = disp_range // 2

        self.tr_ptr = 0
        self.curr_epoch = 0

        self.file_ids = np.fromfile(os.path.join(util_root, 'myPerm.bin'), '<f4')
        self.tr_loc = np.fromfile(('%s/tr_%d_%d_%d.bin') % 
            (util_root, num_tr_img, self.half_patch, self.half_range), '<f4').reshape(-1, 5).astype(int)

        if num_val_img == 0:
            self.val_loc = self.tr_loc
            print('validate on training set..')
        else:
            self.val_loc = np.fromfile(('%s/val_%d_%d_%d.bin') % 
                (util_root, num_val_img, self.half_patch, self.half_range), '<f4').reshape(-1,5).astype(int)

        print(('#training locations: %d -- #valuation locations: %d') % (self.tr_loc.shape[0], 
            self.val_loc.shape[0]))

        for i in range(2, 5):
            self.tr_loc[:, i] -= 1
            self.val_loc[:, i] -= 1


        self.ldata = {}
        self.rdata = {}

  
        self.tr_perm = np.arange(self.tr_loc.shape[0]) 
        self.val_perm = np.arange(self.val_loc.shape[0]) 
        np.random.shuffle(self.tr_perm)
        np.random.shuffle(self.val_perm)


        for i in range(num_tr_img+num_val_img):

            fn = self.file_ids[i]
            if data_version == 'kitti2015':
                l_img = misc.imread(('%s/image_2/%06d_10.png') % (data_root, fn))
                r_img = misc.imread(('%s/image_3/%06d_10.png') % (data_root, fn))
            
            elif data_version == 'kitti2012':
                l_img = misc.imread(('%s/image_0/%06d_10.png') % (data_root, fn))
                r_img = misc.imread(('%s/image_1/%06d_10.png') % (data_root, fn))
         

            l_img = (l_img - l_img.mean()) / l_img.std()
            r_img = (r_img - r_img.mean()) / r_img.std()

            self.ldata[fn] = l_img.reshape(l_img.shape[0], l_img.shape[1], self.num_channels)
            self.rdata[fn] = r_img.reshape(r_img.shape[0], r_img.shape[1], self.num_channels)

        self.batch_left = np.zeros((self.batch_size, self.patch_size, self.patch_size, self.num_channels))
        self.batch_right = np.zeros((self.batch_size, self.patch_size, self.patch_size + self.disp_range - 1, self.num_channels))
        self.batch_label = np.zeros((self.batch_size, disp_range))
        dist = [0.05, 0.2, 0.5, 0.2, 0.05]
        half_dist = len(dist) // 2
        count = 0
        for i in range(disp_range //2 - half_dist, disp_range // 2 + half_dist + 1):
            self.batch_label[:, i] = dist[count]
            count += 1


        self.val_left = np.zeros((num_val_loc, self.patch_size, self.patch_size, self.num_channels))
        self.val_right = np.zeros((num_val_loc, self.patch_size, self.patch_size + self.disp_range - 1, self.num_channels))

        self.val_label = np.zeros((num_val_loc, disp_range))
        count = 0
        for i in range(disp_range //2 - half_dist, disp_range // 2 + half_dist + 1):
            self.val_label[:, i] = dist[count]
            count += 1

        for i in range(num_val_loc):

            img_id, loc_type, center_x, center_y, right_center_x = self.val_loc[self.val_perm[i], 0], self.val_loc[self.val_perm[i], 1], self.val_loc[self.val_perm[i], 2], self.val_loc[self.val_perm[i], 3], self.val_loc[self.val_perm[i], 4]
            right_center_y = center_y


            self.val_left[i] = self.ldata[img_id][(center_y-self.half_patch) : (center_y+self.half_patch + 1), (center_x-self.half_patch) : (center_x+self.half_patch + 1), :]
            if loc_type == 1: # horizontal
                self.val_right[i] = self.rdata[img_id][right_center_y-self.half_patch : right_center_y+self.half_patch + 1, right_center_x-self.half_patch-self.half_range : right_center_x+self.half_patch+self.half_range + 1, :]
            elif loc_type == 2: # vertical
                self.val_right[i] = np.transpose(self.rdata[img_id][right_center_y-self.half_patch-self.half_range : right_center_y+self.half_patch+self.half_range + 1, right_center_x-self.half_patch : right_center_x+self.half_patch + 1, :], (1, 0, 2))

        print('validation created: num(%d)' % num_val_loc)


    def next_batch(self):

        for idx in range(self.batch_size):
            i = self.tr_ptr + (idx + 1)
            if i > self.tr_perm.shape[0]:
                i = 1
                self.tr_ptr = -(idx + 1) + 1
                self.curr_epoch = self.curr_epoch + 1
                print('....epoch id: ' + self.curr_epoch + ' done ......\n')
            i -= 1
            
            img_id, loc_type, center_x, center_y, right_center_x = self.tr_loc[self.tr_perm[i]][0], self.tr_loc[self.tr_perm[i]][1], self.tr_loc[self.tr_perm[i]][2], self.tr_loc[self.tr_perm[i]][3], self.tr_loc[self.tr_perm[i]][4]
            right_center_y = center_y
            
            if loc_type == 1: # horizontal
                self.batch_left[idx] = self.ldata[img_id][(center_y-self.half_patch) : (center_y+self.half_patch + 1), (center_x-self.half_patch) : (center_x+self.half_patch + 1), :]
                self.batch_right[idx] = self.rdata[img_id][right_center_y-self.half_patch : right_center_y+self.half_patch + 1, right_center_x-self.half_patch-self.half_range : right_center_x+self.half_patch+self.half_range + 1, :]
            elif loc_type == 2: # vertical
                self.batch_left[idx] = np.transpose(self.ldata[img_id][(center_y-self.half_patch) : (center_y+self.half_patch + 1), (center_x-self.half_patch) : (center_x+self.half_patch + 1), :], (1, 0, 2))
                self.batch_right[idx] = np.transpose(self.rdata[img_id][right_center_y-self.half_patch-self.half_range : right_center_y+self.half_patch+self.half_range + 1, right_center_x-self.half_patch : right_center_x+self.half_patch + 1, :], (1, 0, 2))
            

        self.tr_ptr = self.tr_ptr + self.batch_size
        return self.batch_left, self.batch_right, self.batch_label


    def evaluate(self):
        return self.val_left, self.val_right, self.val_label

if __name__ == '__main__':
    dh = Data_handler(data_version='kitti2012', data_root=os.path.expanduser('~/stereoMatching/KITTI2012stereo/training'),  
        util_root=os.path.expanduser('~/stereoMatching/preprocess/debug_12'), num_tr_img=160, 
        num_val_img=34, num_val_loc=100, batch_size=128, patch_size=37, disp_range=201)

    bleft, bright, blabels = dh.next_batch()
    print(bleft.shape)
    print(blabels[:5])

