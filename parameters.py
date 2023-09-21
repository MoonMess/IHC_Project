"""
Author: Mounir Messaoudi <mounir.messaoudi45@gmail.com>
10/07/2023
Read IHC slides and extract the cell detection using stardist
"""
import os
# ==================== #
#       PARAMETERS     #
# ==================== #
if __name__ == "__main__":
    print("parameters.py is not supposed to be run as main")
    exit()

if "cluster" in os.getcwd():
    DEFAULT_INPUT_SLIDE_FOLDER = "/cluster/CBIO/data1/mlerousseau/datasets/curie/prosper/original_slides" #cluster file
else:
    DEFAULT_INPUT_SLIDE_FOLDER = "C:/Users/messa/Pictures/IHC/RP" #local file


area_research_detection_level = 4 # level at which the area of interest research is performed (is an index in pyramid : [1,2,4,8,16,32,64,128,256])
nuclei_detection_level = 1  # level at which nuclei detection and classification is performed (is an index in pyramid : [1,2,4,8,16,32,64,128,256])
nuclei_classification_tile_size = 512 # size of the tile used for nuclei classification
threshold_tile_to_keep = 0.01
size_max_cell = 550
size_min_cell = 30
ratio_background_zoom_nuclei_detection_zoom = 2 ** (area_research_detection_level - nuclei_detection_level)
corresponding_tile_size = nuclei_classification_tile_size // ratio_background_zoom_nuclei_detection_zoom
