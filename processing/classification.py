"""
Author: Mounir Messaoudi <mounir.messaoudi45@gmail.com>
10/07/2023
Read IHC slides and extract the cell detection using stardist
"""

# ==================== #
#       MODULES        #
# ==================== #

import numpy as np
import matplotlib.pyplot as plt
from plot.plot import plot_sample, save_classif
import cv2
from skimage.morphology import binary_erosion

# ==================== #
#       FUNCTIONS      #
# ==================== #



def classif_avg(labels, tile_hed, tile, nbr, slide_filename):
    """
    Classify the nuclei in 4 classes based on the average DAB value of the nuclei
    
    Parameters
    ----------
    labels : np.array
        labels of the nuclei
    tile_hed : np.array
        tile in HED color space
    tile : np.array
        tile in RGB color space
    nbr : int
        number of the tile

    Returns
    -------
    class_0 : np.array
        mask of the nuclei classified as class 0
    class_1 : np.array
        mask of the nuclei classified as class 1
    class_2 : np.array  
        mask of the nuclei classified as class 2
    class_3 : np.array
        mask of the nuclei classified as class 3
    merge : np.array    
        mask of the nuclei classified as class 0, 1, 2, 3
    """
    class_0, class_1, class_2, class_3 = np.zeros_like(labels), np.zeros_like(labels), np.zeros_like(labels), np.zeros_like(labels)

    threshold_3 = 0.15 #higher value correspond to class 3
    threshold_2 = 0.115 #higher value correspond to class 2
    threshold_1 = 0.045  #higher value correspond to class 1
    
    nuclei_avg_dab = []
    for nuclei in np.unique(labels):
        if nuclei == 0:
            continue
        #nuclei mask
        current_nuclei_mask = labels == nuclei
        #erosion of the nuclei mask
        current_nuclei_mask_erode = binary_erosion(current_nuclei_mask, np.ones((5,5)))

        hed_value = tile_hed[:,:,2][current_nuclei_mask_erode].mean()
        nuclei_avg_dab.append(hed_value)
        if hed_value > threshold_3:
            class_3[current_nuclei_mask] = nuclei
        elif hed_value > threshold_2:
            class_2[current_nuclei_mask] = nuclei
        elif hed_value > threshold_1:
            class_1[current_nuclei_mask] = nuclei
        else :
            class_0[current_nuclei_mask] = nuclei

    merge = np.zeros_like(labels)
    merge[class_0 != 0] = 1
    merge[class_1 != 0] = 2
    merge[class_2 != 0] = 3
    merge[class_3 != 0] = 4
    
    fig_sample = plot_sample(tile, class_0, class_1, class_2, class_3)
    merge = np.zeros_like(labels)
    merge[class_0 != 0] = 1
    merge[class_1 != 0] = 2
    merge[class_2 != 0] = 3
    merge[class_3 != 0] = 4
    
    #saving the classification plot's
    save_classif(tile, merge, fig_sample, nuclei_avg_dab, threshold_1, threshold_2, threshold_3, nbr, "avg", slide_filename)
    return class_0, class_1, class_2, class_3, merge

def classif_max(labels, tile_hed, tile, nbr, slide_filename):
    """
    Classify the nuclei in 4 classes based on the max DAB value of the nuclei
    
    Parameters
    ----------
    labels : np.array
        labels of the nuclei
    tile_hed : np.array
        tile in HED color space
    tile : np.array
        tile in RGB color space
    nbr : int
        number of the tile

    Returns
    -------
    class_0 : np.array
        mask of the nuclei classified as class 0
    class_1 : np.array
        mask of the nuclei classified as class 1
    class_2 : np.array  
        mask of the nuclei classified as class 2
    class_3 : np.array
        mask of the nuclei classified as class 3
    merge : np.array    
        mask of the nuclei classified as class 0, 1, 2, 3
    """
    labels.copy()
    class_0, class_1, class_2, class_3 = np.zeros_like(labels), np.zeros_like(labels), np.zeros_like(labels), np.zeros_like(labels)

    threshold_3 = 1 #higher value correspond to class 3
    threshold_2 = 0.375 #higher value correspond to class 2
    threshold_1 = 0.1  #higher value correspond to class 1
    
    nuclei_max_dab = []
    for nuclei in np.unique(labels):
        if nuclei == 0:
            continue
        current_nuclei_mask = labels == nuclei
        hed_value_max = tile_hed[:,:,2][current_nuclei_mask].max()
        nuclei_max_dab.append(hed_value_max)
        if hed_value_max > threshold_3:
            class_3[current_nuclei_mask] = nuclei
        elif hed_value_max > threshold_2:
            class_2[current_nuclei_mask] = nuclei
        elif hed_value_max > threshold_1:
            class_1[current_nuclei_mask] = nuclei
        else :
            class_0[current_nuclei_mask] = nuclei
            

    fig_sample = plot_sample(tile, class_0, class_1, class_2, class_3)
    merge = np.zeros_like(labels)
    merge[class_0 != 0] = 1
    merge[class_1 != 0] = 2
    merge[class_2 != 0] = 3
    merge[class_3 != 0] = 4

    #saving the classification plot's
    save_classif(tile, merge, fig_sample, nuclei_max_dab, threshold_1, threshold_2, threshold_3, nbr, "max", slide_filename)

    return class_0, class_1, class_2, class_3, merge

def classif_mix(labels, tile_hed, tile, nbr, slide_filename):
    """
    Classify the nuclei in 4 classes based on the max and avg DAB value of the nuclei

    Parameters
    ----------
    labels : np.array
        labels of the nuclei
    tile_hed : np.array
        tile in HED color space
    tile : np.array
        tile in RGB color space
    nbr : int
        number of the tile

    Returns
    -------
    class_0 : np.array
        mask of the nuclei classified as class 0
    class_1 : np.array
        mask of the nuclei classified as class 1
    class_2 : np.array  
        mask of the nuclei classified as class 2
    class_3 : np.array
        mask of the nuclei classified as class 3
    merge : np.array    
        mask of the nuclei classified as class 0, 1, 2, 3
    """
    class_0, class_1, class_2, class_3 = np.zeros_like(labels), np.zeros_like(labels), np.zeros_like(labels), np.zeros_like(labels)

    threshold_3_avg = 0.15 #higher value correspond to class 3
    threshold_2_avg = 0.115 #higher value correspond to class 2
    threshold_1_avg = 0.045  #higher value correspond to class 1
        
    threshold_3_max = 1 #higher value correspond to class 3
    threshold_2_max = 0.375 #higher value correspond to class 2
    threshold_1_max = 0.1  #higher value correspond to class 1

    nuclei_avg_dab, nuclei_max_dab = [], []
    for nuclei in np.unique(labels):
        if nuclei == 0:
            continue
        #nuclei mask
        current_nuclei_mask = labels == nuclei
        #erosion of the nuclei mask, to improve the avg AND MAX value

        current_nuclei_mask_eroded = binary_erosion(current_nuclei_mask, np.ones((4,4)))
        if current_nuclei_mask_eroded.sum() > 0:
            current_nuclei_mask = current_nuclei_mask_eroded

        avg_hed_value = tile_hed[:,:,2][current_nuclei_mask].mean()
        nuclei_avg_dab.append(avg_hed_value)
        max_hed_value = tile_hed[:,:,2][current_nuclei_mask].max()
        nuclei_max_dab.append(max_hed_value)
        #Class 0
        if avg_hed_value < threshold_1_avg:
            class_0[current_nuclei_mask] = nuclei
        elif max_hed_value < threshold_1_max:
            class_0[current_nuclei_mask] = nuclei
        #Class 1
        elif avg_hed_value < threshold_2_avg:
            class_1[current_nuclei_mask] = nuclei
        #Class 2
        elif avg_hed_value < threshold_3_avg:
            class_2[current_nuclei_mask] = nuclei
        #Class 3
        else :
            class_3[current_nuclei_mask] = nuclei

    merge = np.zeros_like(labels)
    merge[class_0 != 0] = 1
    merge[class_1 != 0] = 2
    merge[class_2 != 0] = 3
    merge[class_3 != 0] = 4
    
    fig_sample = plot_sample(tile, class_0, class_1, class_2, class_3)
    merge = np.zeros_like(labels)
    merge[class_0 != 0] = 1
    merge[class_1 != 0] = 2
    merge[class_2 != 0] = 3
    merge[class_3 != 0] = 4
    
    #saving the classification plot's
    threshold_1 = [threshold_1_avg, threshold_1_max]
    threshold_2 = [threshold_2_avg, threshold_2_max]
    threshold_3 = [threshold_3_avg, threshold_3_max]
    nuclei_dab = [nuclei_avg_dab, nuclei_max_dab]

    save_classif(tile, merge, fig_sample, nuclei_dab, threshold_1, threshold_2, threshold_3, nbr, "mix", slide_filename)
    return class_0, class_1, class_2, class_3, merge