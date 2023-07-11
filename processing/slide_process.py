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

OPENSLIDE_PATH = r'C:/Users/messa/AppData/Local/Programs/Python/Python310/Lib/site-packages/openslide/openslide-win64-20221217/bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from tqdm import tqdm 

from parameters import area_research_detection_level, nuclei_detection_level, ratio_background_zoom_nuclei_detection_zoom, corresponding_tile_size
from processing.tile_process import update_size, filter_enough_zone, get_tile, process_tile
from plot.plot import plot_heatmap

# ==================== #
#       FUNCTIONS      #
# ==================== #

def open_slide(slide_object):
    """
    Open the slide object and return the thumbnail

    Parameters
    ----------
    slide_object : slide object
        Slide object from openslide.
    
    Returns
    -------
    thumbnail_size : tuple(int, int)
        Size of the thumbnail.
    thumbnail : PIL.Image
        Thumbnail.
    """
    #checking downsample level
    assert slide_object.level_downsamples[area_research_detection_level] == int(2 ** area_research_detection_level)
    assert slide_object.level_downsamples[nuclei_detection_level] == int(2 ** nuclei_detection_level)
    assert slide_object.properties['openslide.objective-power'] == '40' #magnification 
    
    #getting thumbnail
    thumbnail_size= slide_object.level_dimensions[area_research_detection_level]
    thumbnail = slide_object.get_thumbnail(thumbnail_size)
    
    return thumbnail_size, thumbnail


def get_slide(slide_folder):
    """
    Return a shuffle list of the IHC file (finishing by .ndpi)
    in the slide folder

    Parameters
    ----------
    slide_folder : string
        Path to the folder containing the IHC slides.

    Returns
    -------
    slide_filenames : List(string)
        List of the IHC slides filenames.

    """
    slide_filenames = os.listdir(slide_folder)
    slide_filenames = list(filter(lambda filename: filename.endswith('.ndpi'), slide_filenames))
    #slide_filenames = list(filter(lambda filename: '669444H_RO' in filename, slide_filenames))
    
    slide_filenames = list(filter(lambda filename: '250001010' in filename, slide_filenames))

    random.shuffle(slide_filenames)
    return slide_filenames



# ==================== #
#         MAIN         #
# ==================== #

def process_slide(slide_filename, slide_folder):
    if not os.path.exists(os.path.join('plot/Classification', slide_filename)):
        os.mkdir(os.path.join('plot/Classification', slide_filename))

    slide_object = openslide.OpenSlide(os.path.join(slide_folder, slide_filename))
    thumbnail_size, thumbnail = open_slide(slide_object)
    
    # Separate the stains from the IHC image
    ihc_rgb = np.asarray(thumbnail)
    ihc_hed = rgb2hed(ihc_rgb)

    # Create an RGB image for each of the stains
    ihc_rgb2 = hed2rgb(np.stack((ihc_hed[:, :, 0], ihc_hed[:, :, 1], ihc_hed[:, :, 2]), axis=-1))

    # Compute a mask
    lum = rgb2gray(ihc_rgb2)
    mask = remove_small_holes(
        remove_small_objects(
            lum < 0.7, 500),
        500)

    mask = opening(mask, disk(3))
    
    mask = update_size(mask,corresponding_tile_size)
    heatmap_mat = np.zeros(thumbnail_size)
    selected_i_and_j = filter_enough_zone(mask, corresponding_tile_size)
    print(slide_filename)
    for nbr, (i, j) in enumerate(tqdm(selected_i_and_j)):
        tile_as_pil = get_tile(slide_object, ratio_background_zoom_nuclei_detection_zoom,  i, j, corresponding_tile_size)
        tile = np.asarray(tile_as_pil)[..., :3]
        tile_hed = rgb2hed(tile)
        if ((tile_hed[:,:,0] > 0.2).sum() <= 0) or ((tile_hed[:,:,2]> 0.1).sum() <= 0) :
            continue
        #save_tile(tile, nbr) #to build test set

        nbr_cells = process_tile(tile, nbr, slide_filename)
        heatmap_mat[j*corresponding_tile_size:(j+1)*corresponding_tile_size,i*corresponding_tile_size:(i+1)*corresponding_tile_size] = nbr_cells
    
    plot_heatmap(heatmap_mat, thumbnail, slide_filename)
    
    
