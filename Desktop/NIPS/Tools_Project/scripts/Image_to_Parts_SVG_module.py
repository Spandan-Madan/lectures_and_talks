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
# exceptions to be raised. Ex: NoContourError



def fit_primitives(im_part,FACTOR):
    '''
    Takes image part, and returns the best approximate polygon information.
    Inputs: im_part: Input image part as PIL image.
            FACTOR: sensitivity parameter for what approximate polygon to fit. If kept lower, higher dimentional polygons are fit.
    outputs: apps: approximate polygons that are identified around closed shapes in the image of the part.
            pos: index of the bet part. So best part would be apps[pos] 
            rects: straight bounding box rectangles fitting these parts. length = len(apps)
            hierarchy: hierarchy of parts (children/parent)
    How it works:
            Starts by finding edges in the image and then finding contours in the edged image.
            For each contour, fits an approximate polygon to the contour.
            Finds the approximate polygon which corresponds to the largest area bounding box rectangle.
            Returns 

    '''
    open_cv_image = np.array(im_part) 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    edged = cv2.Canny(open_cv_image, 30, 200)
    _, contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(cnt) for cnt in contours]
    rects = sorted(rects,key=lambda  x:x[1],reverse=True)

    shapes = []
    apps = []
    if len(contours) < 1:
        print('No contours')
        return -1
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,FACTOR*cv2.arcLength(cnt,True),True)
        if len(approx)==3:
            shapes.append('triange')
            apps.append(approx)
        elif len(approx)==4:
            shapes.append('4 sided')
            apps.append(approx)
        else:
            shapes.append('none')
            apps.append(approx)
    

    # Pick best one of the approximated contours - Right now, max area = best.
    # Alternative to try = Trying the best
    max_area = 0
    for ct in range(len(rects)):
        rect = rects[ct]
        area = rect[2]*rect[3]
        if area > max_area:
            max_area = area
            pos = ct

    return apps,pos,rects,hierarchy


def im_to_svg(impath,save_path,part1=0,part2=1,actual = False):
    '''
    Goal: This function takes in a function, and generates an approximate svg for it.
    Optionally, it can also apply style from a separate icon to the parsed SVG, if another icon is supplied. 
    Scripts it inherits from: (1) icon_helper_functions.py
    Input: impath = path of icon to parse.
            save_path = path where to save the SVG's. This should be a complete file path with .svg extension
    output: prints out SVG to a file, returns nothing.
    Additional functionality that can be accessed - Part saving, uncomment some code.

    How it works: reads image, and depending on which tool it is, it picks a FACTOR value. 
                    Splits image into parts using image_splitter function inherited from icon_helper_functions
                    For each part, finds the primitives using fit_primitives()
                    Then, for each primitive found, creates a corresponding SVG polygon.
                    Writes all polygons together into an SVG file.
    '''

    # Read image as both PIL image and open cv array.
    im = Image.open(impath)
    # imcv = cv2.imread(impath,-1)[:,:,3]

    # Read tool name and get appropriate FACTOR to be used when fitting approximate polygon to parts
    try:
        tool_name = impath.split('/')[3]
    except:
        tool_name = 'dummy'
    factor_dict = {'bricklayer_hammer':0.005,'corkscrew':0.005,'garden_spade':0.02,'claw_hammer':0.02,'crescent_wrench':0.005}
    
    if tool_name in factor_dict.keys():
        FACTOR = factor_dict[tool_name]
    else:
        FACTOR = 0.005
    print('FACTOR is',FACTOR)
    # Invoke helper function to split into parts based on geometry
    if actual == True:
        ims = image_splitter(impath,filled=False,actual=True)
    else:
        ims = image_splitter(impath,filled=True)
    PARTS = (ims[part1],ims[part2])


    best_shapes = []
    
    svg_start = '<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">'
    svg_end = '</svg>'

    # For each part, we get the best shape (NOTE: update definition of best)
    for i in range(len(PARTS)):
        all_shapes,pos,rects,hierarchy = fit_primitives(PARTS[i],FACTOR)
        best_shape = all_shapes[pos]
        best_shapes.append(best_shape)
        
        # Create polygon element for above coordinates
        coordinates = ''
        ct = 0
        for point in best_shape:
            if ct > 0:
                coordinates += ' '
            x,y = point[0][0],point[0][1]
            coordinates += '%s,%s'%(x,y)
            ct += 1
        element = '<polygon points="%s" stroke-linejoin="round" style="stroke:black; stroke-width: 5; fill: white"/>'%coordinates
        
        # create SVG by adding above polygon elements
        if i == 0:
            elements = element
        else:
            elements += '\n' + element
        svg_parts = svg_start + '\n' + element + svg_end
        
        # # Uncomment this block if you want to save parts separately
        # # Save the parts as SVG's separately
        # f = open('Part_%s.svg'%i,'w')
        # print(svg_parts,file = f)
        # f.close()
    
    # Save the SVG approximation for whole icon at given save_path
    svg_content = svg_start + '\n' + elements + svg_end
    save_path
    f = open(save_path,'w')
    print(svg_content,file = f)
    f.close()