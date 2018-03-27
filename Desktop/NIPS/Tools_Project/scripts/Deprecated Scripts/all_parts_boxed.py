import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib import cm
from scipy import ndimage
import sys
import matplotlib.patches as patches

base_path = '/Users/spandanmadan/Desktop/Tools_Project/Icons_clean/'
all_tools = os.listdir(base_path)
final_path = '/Users/spandanmadan/Desktop/Tools_Project/Icons_boxed/'

def splitter(impath,final_path):

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
    try:
        m2 = -1/m1
    except:
        print("couldnt coz m1 is ",m1)

    if m1 < 0.002 or m2 < 0.002:
        exp_fact = 5
    else:
        exp_fact = 0

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
        left1, upper1, right1, lower1 = im1.getbbox()
        left2, upper2, right2, lower2 = im2.getbbox()
    else:
        axis = 2
        left1, upper1, right1, lower1 = im3.getbbox()
        left2, upper2, right2, lower2 = im4.getbbox()

    left1 = left1-exp_fact
    right1 = right1+exp_fact
    upper1 = upper1-exp_fact
    lower1 = lower1+exp_fact

    fig,ax = plt.subplots(1,figsize=(5,5))
    ax.imshow(im); plt.axis('off')

    rect = patches.Rectangle((left1,upper1), right1-left1-1, lower1-upper1-1 , fill = None, edgecolor="g")
    ax.add_patch(rect)

    rect = patches.Rectangle((left2,upper2), right2-left2-1, lower2-upper2-1 , fill = None, edgecolor="b")
    ax.add_patch(rect)
    plt.savefig(output_name_base + "_boxed.jpg")
    plt.close()



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
