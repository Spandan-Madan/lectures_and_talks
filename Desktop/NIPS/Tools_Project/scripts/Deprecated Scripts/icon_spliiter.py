import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib import cm
from scipy import ndimage
import sys

img_path = sys.argv[1]

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

# If you want to plot, uncomment code below. Scales decide start and end of points of the axis lines drawn.
# scale = 80
# scale2 = 2
# scale1 = 2
# implot = plt.imshow(im)
data_total = np.array(im)
# # plt.plot([x_v1*-scale, x_v1*scale],
# #          [y_v1*-scale, y_v1*scale], color='red')
# # plt.plot([x_v2*-scale, x_v2*scale],
# #          [y_v2*-scale, y_v2*scale], color='blue')
# plt.plot([ctx-ctx/scale1, ctx+ctx/scale1],
#          [cty-ctx/scale1*m1, cty+ctx/scale1*m1], color='green')
# plt.plot([ctx+ctx/scale2, ctx-ctx/scale2],
#          [cty+ctx/scale2*m2, cty-ctx/scale2*m2], color='blue')
# # plt.plot(x, y, 'k.')
# plt.axis('equal')
# # plt.gca().invert_yaxis()  # Match the image system with origin at top left
# plt.show()


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
print(name,ext)
outdir = sys.argv[2]
output_name_base = outdir + '/' + name

if im1.mode != 'RGB':
    im1 = im1.convert('RGB')
if im2.mode != 'RGB':
    im2 = im2.convert('RGB')
if im3.mode != 'RGB':
    im3 = im3.convert('RGB')
if im4.mode != 'RGB':
    im4 = im4.convert('RGB')

im1.save(output_name_base + "_part1." + ext)
im2.save(output_name_base + "_part2." + ext)
im3.save(output_name_base + "_part3." + ext)
im4.save(output_name_base + "_part4." + ext)
