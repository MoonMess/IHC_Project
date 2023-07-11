"""
Author: Mounir Messaoudi <mounir.messaoudi45@gmail.com>
10/07/2023
Read IHC slides and extract the cell detection using stardist
"""

# ==================== #
#       MODULES        #
# ==================== #
import os
import random
import numpy as np
from skimage.color import rgb2hed, hed2rgb, rgb2gray
from skimage.morphology import remove_small_holes, remove_small_objects, opening, disk
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
from processing.classification import classif_avg, classif_max, classif_mix
from parameters import area_research_detection_level, nuclei_detection_level, threshold_tile_to_keep, nuclei_classification_tile_size, size_max_cell, size_min_cell
# ==================== #
#       PARAMETERS     #
# ==================== #

model_stardist = StarDist2D.from_pretrained('2D_versatile_he')

# ==================== #
#       FUNCTIONS      #
# ==================== #

def filter_enough_zone(mask, corresponding_tile_size):
    #keeping the interesting part : with enough yellow TO IMPROVE
    indexes_with_enough_yellow_material = mask.reshape(
        (mask.shape[0] // corresponding_tile_size, corresponding_tile_size,
         mask.shape[1] // corresponding_tile_size, corresponding_tile_size)).\
        sum(3).sum(1) > (corresponding_tile_size * corresponding_tile_size * threshold_tile_to_keep) 
    
    selected_i_and_j = np.nonzero(indexes_with_enough_yellow_material)
    selected_i_and_j = list(zip(selected_i_and_j[0], selected_i_and_j[1]))
    random.shuffle(selected_i_and_j)
    print(np.array(selected_i_and_j).shape)
    return selected_i_and_j

def update_size(mask,corresponding_tile_size):
    # adapting the sizes, as we are changing of scale
    if mask.shape[0] % corresponding_tile_size != 0:
        mask = np.concatenate(
            (mask,
             np.zeros((corresponding_tile_size -
                      (mask.shape[0] % corresponding_tile_size),
                      mask.shape[1]),
                      dtype=mask.dtype)), axis=0)
    if mask.shape[1] % corresponding_tile_size != 0:
        mask = np.concatenate(
            (mask,
             np.zeros((mask.shape[0],
                       corresponding_tile_size -
                       (mask.shape[1] % corresponding_tile_size)),
                      dtype=mask.dtype)), axis=1)
    return mask

def get_tile(slide_object, ratio_background_zoom_nuclei_detection_zoom,  i, j, corresponding_tile_size):
    # tile_mask = mask[
    #                         i * corresponding_tile_size:(i + 1) * corresponding_tile_size,
    #                         j * corresponding_tile_size:(j + 1) * corresponding_tile_size]
    # assert tile_mask.sum() > \
    #     (corresponding_tile_size * corresponding_tile_size * threshold_tile_to_keep)


    tile = slide_object.read_region(
            (j * corresponding_tile_size * ratio_background_zoom_nuclei_detection_zoom * 2 ** nuclei_detection_level,
             i * corresponding_tile_size * ratio_background_zoom_nuclei_detection_zoom * 2 ** nuclei_detection_level),
            nuclei_detection_level, (nuclei_classification_tile_size, nuclei_classification_tile_size))

    return tile#, tile_mask

def filtering_labels(labels):
    """
    Remove labels that are too small or too large
    
    Parameters
    ----------
    labels : np.array
        labels of the nuclei
        
    Returns
    -------
    labels : np.array
        labels of the nuclei after filtering
    """
    filtered_labels = labels.copy()
    size = []
    for nuclei in np.unique(filtered_labels):
        if nuclei == 0:
            continue
        current_nuclei_mask = filtered_labels == nuclei
        # remove nuclei too large prediction
        size_nuclei = current_nuclei_mask.sum()

        too_big = (size_nuclei > size_max_cell)
        too_small = (size_nuclei < size_min_cell)
        if (too_big or too_small): 
            filtered_labels[current_nuclei_mask] = 0
            continue
        size.append(int(size_nuclei))
    return filtered_labels

# ==================== #
#         MAIN         #
# ==================== #

def process_tile(tile_as_pil, nbr, slide_filename):
    tile = np.asarray(tile_as_pil)[..., :3]
    tile_hed = rgb2hed(tile)

    labels, _ = model_stardist.predict_instances(normalize(tile), prob_thresh=0.3)    
    
    filtered_labels = filtering_labels(labels)

    if len(np.unique(filtered_labels)) > 1 :
        class_0, class_1, class_2, class_3, class_merge = classif_max(filtered_labels, tile_hed, tile, nbr, slide_filename)
        class_0, class_1, class_2, class_3, class_merge = classif_avg(filtered_labels, tile_hed, tile, nbr, slide_filename)
        class_0, class_1, class_2, class_3, class_merge = classif_mix(filtered_labels, tile_hed, tile, nbr, slide_filename)

    return len(np.unique(filtered_labels))