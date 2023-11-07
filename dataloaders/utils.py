import matplotlib.pyplot as plt
import numpy as np
import torch

def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    
    if dataset == 'spacenet':
        n_classes = 2
        label_colours = get_spacenet_labels()
    elif dataset == 'DeepGlobe':
        n_classes = 2
        label_colours = get_deepglobe_labels()
    elif dataset == '7b52009b64fd0a2a49e6d8a939753077792b0554':
        n_classes = 2
        label_colours = get_tia_labels()    
    elif dataset == '0ade7c2cf97f75d009975f4d720d1fa6c19f4897':
        n_classes = 2
        label_colours = get_tia_labels()    
    elif dataset == '1b6453892473a467d07372d45eb05abc2031647a':
        n_classes = 2
        label_colours = get_tia_labels()    
    elif dataset == '17ba0791499db908433b80f37c5fbc89b870084b':
        n_classes = 2
        label_colours = get_tia_labels()    
    elif dataset == '356a192b7913b04c54574d18c28d46e6395428ab':
        n_classes = 2
        label_colours = get_tia_labels()    
    elif dataset == '902ba3cda1883801594b6e1b452790cc53948fda':
        n_classes = 2
        label_colours = get_tia_labels()    
    elif dataset == 'b1d5781111d84f7b3fe45a0852e59758cd7a87e5':
        n_classes = 2
        label_colours = get_tia_labels()    
    elif dataset == 'bd307a3ec329e10a2cff8fb87480823da114f8f4':
        n_classes = 2
        label_colours = get_tia_labels()    
    elif dataset == 'c1dfd96eea8cc2b62785275bca38ac261256e278':
        n_classes = 2
        label_colours = get_tia_labels()    
    elif dataset == 'fa35e192121eabf3dabf9f5ea6abdbcbc107ac3b':
        n_classes = 2
        label_colours = get_tia_labels()    
    elif dataset == 'mergedataset':
        n_classes = 2
        label_colours = get_tia_labels()    
    
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def get_spacenet_labels():
    """Load the mapping that associates spacenet classes with label colors
    Returns:
        np.ndarray with dimensions (2, 3)
    """
    return np.asarray([[0, 0, 0], [255, 255, 255]])

def get_deepglobe_labels():
    """Load the mapping that associates deepglobe classes with label colors
    Returns:
        np.ndarray with dimensions (2, 3)
    """
    return np.asarray([[0, 0, 0], [255, 255, 255]])

def get_tia_labels():
    """Load the mapping that associates deepglobe classes with label colors
    Returns:
        np.ndarray with dimensions (2, 3)
    """
    return np.asarray([[0, 0, 0], [255, 255, 255]])

# def get_labels(num_classes):
#     """Load the mapping that associates classes with label colors
#     Args:
#         num_classes (int): Number of classes in the dataset
#     Returns:
#         np.ndarray with dimensions (num_classes, 3)
#     """
#     label_colours = np.random.randint(0, 256, size=(num_classes, 3))
#     return label_colours
