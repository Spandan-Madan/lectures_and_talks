import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib import cm
from scipy import ndimage
import sys

base_path = '/Users/spandanmadan/Desktop/Tools_Project/Icons_clean/'
all_tools = os.listdir(base_path)
final_path = '/Users/spandanmadan/Desktop/Tools_Project/Icons_parts/'

def splitter(impath,final_path=False):

    img_path = impath

    im = Image.open(img_path).convert('LA').resize((200,200))
    data = np.array(im)
    dat = data[:,:,1]


    #Convert into binary image
    for i in range(200):
        for j in range(200):
            if dat[i][j]>0:
                dat[i][j] = 1

    # Measure total nonzero points
    point_ct = 0
    for i in range(200):
        for j in range(200):
            if dat[i][j]>0:
                point_ct += 1

    # Store non zero point locations into numpy array
    points = np.zeros((point_ct,2))
    point_ct = 0
    for i in range(200):
        for j in range(200):
            if dat[i][j]>0:
                points[point_ct][0] = j
                points[point_ct][1] = i
                point_ct += 1

    y,x = points[:,0],points[:,1]
    scale_size = max(y) - min(y)

    # Normalize coordinates and calculate eigenvectors of covariance matrix

    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x, y])
    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)


    # Get principal components using eigenvalues
    sort_indices = np.argsort(evals)[::-1]
    evec1, evec2 = evecs[:, sort_indices]
    x_v1, y_v1 = evec1  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evec2

    # Get slopes of the two principal axes
    slope = y_v1/x_v1
    m1 = slope
    m2 = -1/m1

    # Get center of mass of image
    blobs = dat > 0
    labels, nlabels = ndimage.label(blobs)
    # find the center of mass of each label
    t = ndimage.center_of_mass(dat, labels, np.arange(nlabels) + 1 )
    # calc sum of each label, this gives the number of pixels belonging to the blob
    s = ndimage.sum(blobs, labels,  np.arange(nlabels) + 1 )
    # notation of output (y,x)
    cty,ctx = t[s.argmax()]

    # Split image based on axis
    data_total = np.array(im)
    img_data = data_total[:,:,1]
    imdat = Image.fromarray(img_data)

    dat1 = np.zeros((200,200))
    dat2 = np.zeros((200,200))
    dat3 = np.zeros((200,200))
    dat4 = np.zeros((200,200))
    # dataa = img
    for i in range(200):
        for j in range(200):
            y = i
            x = j
            val = img_data[i][j]
            eqn1 = y+m1*ctx-m1*x-cty
            eqn2 = y+m2*ctx-m2*x-cty
            if eqn1 > 0:
                dat1[i][j] = val
            else:
                dat2[i][j] = val

            if eqn2 > 0:
                dat3[i][j] = val
            else:
                dat4[i][j] = val


    # In[156]:

    im1 = Image.fromarray(dat1)
    im2 = Image.fromarray(dat2)
    im3 = Image.fromarray(dat3)
    im4 = Image.fromarray(dat4)

    # save parts
    imname = img_path.split('/')[-1]
    name,ext = imname.split('.')
    outdir = final_path
    output_name_base = outdir + '/' + name

    if im1.mode != 'RGB':
        im1 = im1.convert('RGB')
    if im2.mode != 'RGB':
        im2 = im2.convert('RGB')
    if im3.mode != 'RGB':
        im3 = im3.convert('RGB')
    if im4.mode != 'RGB':
        im4 = im4.convert('RGB')

    sum1 = sum(sum(dat1))
    sum2 = sum(sum(dat2))
    sum3 = sum(sum(dat3))
    sum4 = sum(sum(dat4))
    diff1 = abs(sum1-sum2)
    diff2 = abs(sum3-sum4)

    if diff2 < diff1:
        axis = 1
        im1.save(output_name_base + "_part1." + ext)
        im2.save(output_name_base + "_part2." + ext)
    else:
        axis = 2
        im3.save(output_name_base + "_part1." + ext)
        im4.save(output_name_base + "_part2." + ext)

    dat5 = np.zeros((200,200))
    dat6 = np.zeros((200,200))
    dat7 = np.zeros((200,200))
    dat8 = np.zeros((200,200))
    dat9 = np.zeros((200,200))
    dat10 = np.zeros((200,200))

    if axis == 1:
        slope_correct = m1
        slope_other = m2
    else:
        slope_correct = m2
        slope_other = m1
    # dataa = img
    for i in range(200):
        for j in range(200):
            y = i
            x = j
            val = img_data[i][j]
            A = scale_size/10
            B = scale_size/5
            C = -scale_size/10
            eqn3 = y+slope_correct*(ctx+A)-slope_correct*x-(cty+slope_other*A)
            eqn4 = y+slope_correct*(ctx+B)-slope_correct*x-(cty+slope_other*B)
            eqn5 = y+slope_correct*(ctx+C)-slope_correct*x-(cty+slope_other*C)

            if eqn3 > 0:
                dat5[i][j] = val
            else:
                dat6[i][j] = val

            if eqn4 > 0:
                dat7[i][j] = val
            else:
                dat8[i][j] = val

            if eqn5 > 0:
                dat9[i][j] = val
            else:
                dat10[i][j] = val

    im5 = Image.fromarray(dat5).convert('RGB')
    im6 = Image.fromarray(dat6).convert('RGB')
    im7 = Image.fromarray(dat7).convert('RGB')
    im8 = Image.fromarray(dat8).convert('RGB')
    im9 = Image.fromarray(dat9).convert('RGB')
    im10 = Image.fromarray(dat10).convert('RGB')

    # Save Crops
    im5.save(output_name_base + "_part3." + ext)
    im6.save(output_name_base + "_part4." + ext)
    im7.save(output_name_base + "_part5." + ext)
    im8.save(output_name_base + "_part6." + ext)
    im9.save(output_name_base + "_part7." + ext)
    im10.save(output_name_base + "_part8." + ext)

for tool in all_tools:
    if tool.startswith('.'):
        continue

    print("Working on %s\n"%tool)
    tool_folder = base_path + tool
    if os.path.isdir(final_path + "/" + tool):
        pass
    else:
        os.mkdir(final_path + "/" + tool)
    print(tool_folder)
    all_im_names = os.listdir(tool_folder)
    print(len(all_im_names))

    for im in all_im_names:
        if im.startswith('.'):
            continue
        image_path = tool_folder + "/" + im
        print(image_path)
        splitter(image_path,final_path + "/" + tool)
