import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys
sys.path.append('../scripts/')
import cv2
import matplotlib.patches as patches
import math
import scipy
from scipy.ndimage.morphology import binary_closing
from scipy import ndimage
from IPython.display import SVG 
from IPython.display import Pretty 
import pprint
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from icon_helper_functions import *

## UNDER DEVELOPMENT
# More kinds of style beyond just dominant color
# Picking Texture somehow?
# Picking order better?

def apply_style(texture_impath,SVG_path,SVG_stylized_path,tool_name):
    '''
    Takes an SVG file and an icon image path and applies the style of the icon to the svg file.
    Inputs: SVG_path: SVG file path to be opened and edited. This should be the complete file path with .svg extension.
            texture_impath: path of the icon from which style is to be identified.
            SVG_stylized_path: path of file to save
    Outputs: Edits the SVG file in a new file
            Returns nothing

    How it works:
        Reads given icon at path, and splits it into parts.
        For each part, identifies the dominant color.
        Applies dominant colors to the polygons in the SVG using fill: property.
        Right now, order is just sequential, i.e. color from the first part of the icon is applied to color of first polygon
        Good paths to play with - '../data/Flat_icon/hammer/139273.png'
                                    '../data/Flat_icon/hammer/222586.png'
    '''

    # Open image as PIL, convert to open cv and also get parts using image_spliter()
    texture_im = Image.open(texture_impath).resize((200,200))
    imcv = pil_to_cv(texture_im)
    parts = image_splitter(texture_im)


    # For each part, get a mask. Apply this mask to the original image to get colored image of part.
    # For part image, get dominant colors.
    # Dominant color code adapted from here -https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097?gi=a0b763086a39
    part_colors = []
    for part in parts[:2]:
        mask = np.array(part)[:,:,2]
        mask[mask>0] = 255

        res = cv2.bitwise_and(imcv,imcv,mask = mask)
        hist,color,bar = dominant_colors(texture_impath,mask)
        best_color = color[np.argmax(np.asarray(hist))]
        part_colors.append(best_color)


    # If corkscrew, switch colors between parts. Just a hack, will improve later.
    if tool_name == 'corkscrew':
        part_colors.reverse()

    # Open original SVG
    f=open(SVG_path,'r')
    content = f.readlines()
    f.close()

    # Open and create new stylized SVG
    f = open(SVG_stylized_path,'w')
    ct = 0
    for c in content:
        c = c.rstrip()
        if c.startswith('<polygon'):
            R,G,B = part_colors[ct]
            color_string = 'rgb(%s,%s,%s)'%(int(R),int(G),int(B))
            c = c.replace("white",color_string)
            print(c,file = f)
            ct += 1
        else:
            print(c,file=f)
    f.close()