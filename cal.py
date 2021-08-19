import os
import cv2
import numpy as np

path = './results/gt/'
files = os.listdir(path)

error = 0
count = 0
for file in files:
    pt = file.split('_')

    disp_targets = cv2.imread(path + file,cv2.IMREAD_GRAYSCALE)
    predicted_map = cv2.imread('./results/nyu_post/' + pt[1] + '_10.png', cv2.IMREAD_GRAYSCALE)

    if disp_targets.shape == predicted_map.shape:
        valid_gt_pixels = (disp_targets != 0).astype('float')
        
        masked_prediction_valid = predicted_map * valid_gt_pixels

        num_valid_gt_pixels = valid_gt_pixels.sum()

        # NOTE: Use 3-pixel error metric for now.
        num_error_pixels = (np.abs(masked_prediction_valid - disp_targets) > 3).sum()
        
        error_id = num_error_pixels / num_valid_gt_pixels

        error += error_id

        error_str = ('%06d') % int(pt[1]) + '_10' + '\t' + ('%f') % error_id

        print(error_str)
    
    count += 1

print("Mean Error: ", error / count)


