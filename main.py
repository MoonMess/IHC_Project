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
OPENSLIDE_PATH = r'C:/Users/messa/AppData/Local/Programs/Python/Python310/Lib/site-packages/openslide/openslide-win64-20221217/bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
#import from other files
from processing.slide_process import get_slide, process_slide
from parameters import DEFAULT_INPUT_SLIDE_FOLDER
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
    for slide_filename in tqdm(slide_filenames):
        process_slide(slide_filename, slide_folder)
if __name__ == "__main__":
    main()