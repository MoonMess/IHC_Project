# Author: Mounir Messaoudi <mounir.messaoudi45@gmail.com>
#10/07/2023
"""
Read IHC slides and extract the cell detection using stardist
"""

# ==================== #
#       MODULES        #
# ==================== #
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
import os
import numpy as np
import cv2
import random 
from PIL import Image
from matplotlib import gridspec
#import from other files
from parameters import nuclei_classification_tile_size


# ==================== #
#       FUNCTIONS      #
# ==================== #

def mapping_color(labels) : 
        mapping = list(range(len(np.unique(labels)), len(np.unique(labels))*2))
        random.shuffle(mapping)
        copy_labels = labels.copy()
        for n, nuclei in enumerate(np.unique(labels)):
            if nuclei == 0:
                continue
            copy_labels[labels == nuclei] = mapping[n-1] #n-1 because 0 is not a nuclei?
        return copy_labels

def plot_heatmap(heatmap_mat, thumbnail, slide_filename):
    """
    Save a heatmap plot of the slide

    Parameters
    ----------
    heatmap_mat : np.array
        heatmap matrix
    thumbnail : np.array
        thumbnail of the slide
    slide_filename : str
        slide filename

    Returns
    -------
    None.
    """

    brg = mpl.colormaps.get_cmap('brg')
    newcolors = brg(np.linspace(0, 0.5, 256))
    newcmp = ListedColormap(newcolors)
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(thumbnail)
    divider = make_axes_locatable(ax)
    colorbar_axes = divider.append_axes("right",
                                    size="5%",
                                    pad=0.05)

    plt.colorbar(ax.matshow(heatmap_mat.T, alpha=0.5, cmap=newcmp), cax=colorbar_axes)
    plt.savefig(os.path.join('plot/'+slide_filename+'_heatmap.png'), dpi=1200, bbox_inches='tight')
    plt.close()



def compute_cell_focus(tile, current_nuclei_mask, size_crop):
    """
    Compute the focus of the cell, based on the center of mass of the nuclei

    Parameters
    ----------
    tile : np.array
        tile of the slide
    current_nuclei_mask : np.array
        mask of the nuclei
    size_crop : int
        size of the crop

    Returns
    -------
    np.array
        croped cell

    """

    M = cv2.moments((current_nuclei_mask).astype(np.uint8))
    x, y = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]) 
    #coordinate on the patch
    if x < size_crop/2:
        x1,x2 = 0,size_crop
    elif x > nuclei_classification_tile_size-1- size_crop/2:
        x1,x2 = nuclei_classification_tile_size-1-size_crop, nuclei_classification_tile_size-1
    else:
        x1,x2 = x-size_crop/2, x+size_crop/2
    if y < size_crop/2:
        y1,y2 = 0,size_crop
    elif y > nuclei_classification_tile_size-1-size_crop/2:
        y1,y2 = nuclei_classification_tile_size-1-size_crop, nuclei_classification_tile_size-1
    else:
        y1,y2 = y-size_crop/2, y+size_crop/2

    #add the croped cell to the frame
    img = cv2.resize(tile[int(y1):int(y2), int(x1):int(x2)], dsize=(size_crop*2, size_crop*2), interpolation=cv2.INTER_CUBIC)
    
    return img

def plot_sample(tile, class_0, class_1, class_2, class_3):
    """
    Plot a sample of each class

    Parameters
    ----------
    tile : np.array
        tile of the slide
    class_0 : np.array
        class 0
    class_1 : np.array
        class 1
    class_2 : np.array  
        class 2
    class_3 : np.array
        class 3

    Returns
    -------
    np.array
        frame of the sample

    
    """
    margin_between_class = 3 #margin to add between each class
    size_frame = 600 # basic frame without margin between class (without margin)
    size_per_class = size_frame//4 #size of the frame for each class
    size_crop = 25 #size of the crop on the patch(=tile) to focus on the cell (size_crop x size_crop)

    #create a black frame at right size
    #frame_classes = cv2.resize(np.zeros_like(tile), dsize=(size_frame, size_frame + margin_between_class*4), interpolation=cv2.INTER_CUBIC)
    frame_classes = np.ones((size_frame + margin_between_class*4, size_frame, 3), dtype=np.uint8)*255

    labels_0 = np.delete(np.unique(class_0), 0)
    labels_1 = np.delete(np.unique(class_1), 0)
    labels_2 = np.delete(np.unique(class_2), 0)
    labels_3 = np.delete(np.unique(class_3), 0)
    #building the frame, with the margin between each class and the croped cell
    
    for i in range(0, size_per_class -1, size_crop*2):
        for j in range(0, frame_classes.shape[1]-1, size_crop*2):

            if len(labels_0) != 0:
                current_nuclei_mask = class_0 == labels_0[0]
                #Computing center of the cell
                img = compute_cell_focus(tile, current_nuclei_mask, size_crop)
                #add the croped cell to the frame
                frame_classes[i:i+size_crop*2, j:j+size_crop*2] = img
                #remove the label from the list
                labels_0 = np.delete(labels_0, 0)

            if len(labels_1) != 0:
                current_nuclei_mask = class_1 == labels_1[0]
                #Computing center of the cell
                img = compute_cell_focus(tile, current_nuclei_mask, size_crop)
                #add the croped cell to the frame
                frame_classes[i + (size_per_class+margin_between_class):i+ (size_per_class+margin_between_class) +size_crop*2, j:j+size_crop*2] = img
                #remove the label from the list
                labels_1 = np.delete(labels_1, 0)
            if len(labels_2) != 0:
                current_nuclei_mask = class_2 == labels_2[0]
                #Computing center of the cell
                img = compute_cell_focus(tile, current_nuclei_mask, size_crop)
                #add the croped cell to the frame
                frame_classes[i+(size_per_class+margin_between_class)*2:i+(size_per_class+margin_between_class)*2 +size_crop*2, j:j+size_crop*2] = img  
                #remove the label from the list
                labels_2 = np.delete(labels_2, 0)
            if len(labels_3) != 0:
                current_nuclei_mask = class_3 == labels_3[0]
                #Computing center of the cell
                img = compute_cell_focus(tile, current_nuclei_mask, size_crop)
                #add the croped cell to the frame
                frame_classes[i+(size_per_class+margin_between_class)*3:i+(size_per_class+margin_between_class)*3+size_crop*2, j:j+size_crop*2] = img
                #remove the label from the list
                labels_3 = np.delete(labels_3, 0)
    # plt.imshow(frame_classes)
    # plt.axis('off')
    # plt.show()
    return frame_classes

def save_tile(tile, nbr) :    
    """
    Save the tile

    Parameters
    ----------
    tile : np.array
        tile of the slide
    nbr : int
        corresponding number of the tile

    """

    im = Image.fromarray(tile)
    im.save("tiles_for_training/tile_"+str(nbr)+".png")
    im.close()

def save_classif(tile, classification, fig_sample, nuclei_dab, threshold_1, threshold_2, threshold_3, nbr, title, slide_filename) :
    """
    Save the classification

    Parameters
    ----------
    tile : np.array
        tile of the slide
    classification : np.array
        classification of the tile
    fig_sample : np.array
        sample of the classification
    nuclei_dab : np.array
        level of dab, max or avg ro both

    """
    if title == "mix":
        fig = plt.figure(figsize=(10, 10))
        #plt.subplots_adjust(wspace= 0.125, hspace= 0.125)
        gs = gridspec.GridSpec(4, 2)
        ax0 = plt.subplot(gs[0:2,0])
        ax0.imshow(tile)
        ax0.set_title('Patch')
        ax0.axis('off')

        ax1 = plt.subplot(gs[0:2,1])
        ax1.imshow(classification)
        ax1.set_title('Nuclei Classification')
        ax1.axis('off')

        ax2 = plt.subplot(gs[2:4,0])
        ax2.imshow(fig_sample)
        ax2.set_title('Sample Nuclei Classification')
        ax2.axis('off')

        ax3 = plt.subplot(gs[2:3,1])
        ax3.hist(nuclei_dab[0], bins=100, alpha=0.5, label='avg', color='green')
        ax3.plot([threshold_1[0], threshold_1[0]], [0, 10], color='blue', linestyle='dashed', linewidth=1)
        ax3.plot([threshold_2[0], threshold_2[0]], [0, 10], color='brown', linestyle='dashed', linewidth=1)
        ax3.plot([threshold_3[0], threshold_3[0]], [0, 10], color='black', linestyle='dashed', linewidth=1)
        ax3.set_title('Avg DAB per nuclei')

        ax4 = plt.subplot(gs[3:4,1])
        ax4.hist(nuclei_dab[1], bins=100, alpha=0.5, label='max', color='blue')
        ax4.plot([threshold_1[1], threshold_1[1]], [0, 10], color='blue', linestyle='dashed', linewidth=1)
        #ax4.plot([threshold_2[1], threshold_2[1]], [0, 10], color='blue', linestyle='dashed', linewidth=1)
        #ax4.plot([threshold_3[1], threshold_3[1]], [0, 10], color='blue', linestyle='dashed', linewidth=1)
        ax4.set_title('Max DAB per nuclei', )

    else:
        fig, ax = plt.subplots(2,2, figsize=(10, 10))
        ax[0][0].imshow(tile)
        ax[0][0].set_title('Patch')
        ax[0][0].axis('off')

        ax[0][1].imshow(classification)
        ax[0][1].set_title('Nuclei Classification')
        ax[0][1].axis('off')

        ax[1][0].imshow(fig_sample)
        ax[1][0].set_title('Sample Nuclei Classification')
        ax[1][0].axis('off')
        
        if title == "avg":
            ax[1][1].hist(nuclei_dab, bins=100, color='green')
            ax[1][1].plot([threshold_3, threshold_3],[0, 10], linestyle='--', c='blue')
            ax[1][1].plot([threshold_2, threshold_2],[0, 10], linestyle='--', c='brown')
            ax[1][1].plot([threshold_1, threshold_1],[0, 10], linestyle='--', c='black')
            ax[1][1].set_title('Histogram of {} DAB per cell'.format(title))
        else:
            ax[1][1].hist(nuclei_dab, bins=100, color='blue')
            ax[1][1].plot([threshold_3, threshold_3],[0, 10], linestyle='--', c='blue')
            ax[1][1].plot([threshold_2, threshold_2],[0, 10], linestyle='--', c='brown')
            ax[1][1].plot([threshold_1, threshold_1],[0, 10], linestyle='--', c='black')
            ax[1][1].set_title('Histogram of {} DAB per cell'.format(title))
    
    plt.tight_layout()  
    plt.savefig('plot/Classification/{}/Classification_{}_{}.png'.format(slide_filename, nbr, title))
    #plt.show()
    plt.close()