"""
Author: Mounir Messaoudi <mounir.messaoudi45@gmail.com>
10/07/2023
Read IHC slides and extract the cell detection using stardist
"""

# ==================== #
#       MODULES        #
# ==================== #
import argparse
import os 

from multiprocessing import Pool
OPENSLIDE_PATH = r'C:/Users/messa/AppData/Local/Programs/Python/Python310/Lib/site-packages/openslide/openslide-win64-20221217/bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

import numpy as np
from skimage.color import rgb2hed

import itertools

#import from other files
from plot.plot import *
from processing.slide_process import *
from parameters import DEFAULT_INPUT_SLIDE_FOLDER, area_research_detection_level, nuclei_detection_level, nuclei_classification_tile_size, ratio_background_zoom_nuclei_detection_zoom, corresponding_tile_size
from tqdm import tqdm
# ==================== #
#       FUNCTIONS      #
# ==================== #

def get_parser_args(): 
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-s", "--slide_folder", default=DEFAULT_INPUT_SLIDE_FOLDER,
                        help="Path to the input data",)
    args = parser.parse_args()
    return args

# ==================== #
#         MAIN         #
# ==================== #

def main():
    args = get_parser_args()
    slide_folder = args.slide_folder
    slide_filenames = get_slide(slide_folder)
    print(slide_filenames)

    for  slide_filename in tqdm(slide_filenames):
        process_slide(slide_filename, slide_folder)
    #TODO: parallelize
    #with Pool(3) as pool:
        #check what chunksize is
        #pool.starmap(process_slide, zip(slide_filenames, itertools.repeat(slide_folder)))
            
        #mask, slide_object, thumbnail_size, thumbnail, slide_filename = process_slide(slide_filenames[0], slide_folder)

            # ratio_background_zoom_nuclei_detection_zoom = 2 ** (area_research_detection_level - nuclei_detection_level)
            # corresponding_tile_size = nuclei_classification_tile_size // ratio_background_zoom_nuclei_detection_zoom

            # mask = update_size(mask,corresponding_tile_size)

            # selected_i_and_j = filter_enough_zone(mask, corresponding_tile_size)
            #heatmap_mat = np.zeros(thumbnail_size)

            # for nbr, (i, j) in enumerate(tqdm(selected_i_and_j)):
            #         tile_as_pil = get_tile(slide_object, ratio_background_zoom_nuclei_detection_zoom,  i, j, corresponding_tile_size)
            #         tile = np.asarray(tile_as_pil)[..., :3]
            #         tile_hed = rgb2hed(tile)
            #         if ((tile_hed[:,:,0] > 0.2).sum() <= 0) or ((tile_hed[:,:,2]> 0.1).sum() <= 0) :
            #             continue
            #         #save_tile(tile, nbr) #to build test set

            #         nbr_cells = process_tile(tile, nbr, slide_filename)
            #         heatmap_mat[j*corresponding_tile_size:(j+1)*corresponding_tile_size,i*corresponding_tile_size:(i+1)*corresponding_tile_size] = nbr_cells

            # plot_heatmap(heatmap_mat, thumbnail, slide_filename)

if __name__ == "__main__":
    main()