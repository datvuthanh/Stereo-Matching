import matplotlib.pyplot as plt
import numpy as np


def plot_pairwise_training_batches(x, y):
    stack_divider = np.zeros((x.shape[0], x.shape[1], 10, x.shape[-1]))
    y[:, :, 98:102, :] = 0
    stacked_images = np.vstack(
        np.concatenate(
            (x, stack_divider, y),
            axis=2)
        )

    plt.imshow(stacked_images)
