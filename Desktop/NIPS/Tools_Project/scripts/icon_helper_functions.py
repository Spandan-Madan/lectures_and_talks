import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib import cm
from scipy import ndimage
import sys
import scipy
from scipy.ndimage.morphology import binary_closing
import cv2
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import PIL
from skimage import util
########## GENERAL PURPOSE FUNCTIONS ###############


def pil_to_cv(pil_image):
	im_no_alpha = Image.new("RGB", pil_image.size, (255, 255, 255))
	im_no_alpha.paste(pil_image, mask=pil_image.split()[3]) # 3 is the alpha channel
	imcv = np.array(im_no_alpha)
	return imcv
    # # open_cv_image = np.array(pil_image) 
    # # # Convert RGB to BGR 
    # # open_cv_image = open_cv_image[:, :, ::-1].copy() 
    # # return open_cv_image
    # return imcv


def show(img_path,title=False):
	if type(img_path) == Image.Image or type(img_path) == PIL.PngImagePlugin.PngImageFile:
		im = img_path
	else:
		im = Image.open(img_path).resize((200,200))

	plt.axis('off')
	plt.imshow(im)
	if title:
		plt.title(title)
	plt.show()

def fill_im(imcv):
    closed = ndimage.binary_closing(imcv).astype(float)
    return ndimage.binary_fill_holes(closed).astype(float)	


######## IMAGE SPLITTING ##############

def image_splitter(img_path,output_path = False,filled=False,actual = False):
	'''
	Input: argument 1 - image path or PIL Image to be split
		   argument 2 - output path of folder where these images are saved. If nothing provided, images arent saved.
		   argument 3 - if filled = True, image is first filled and then split. If true, pass image path.
	Output: 8 part images as PIL image objects. If output path is provided, the detected parts are saved as images
	'''
	if type(img_path) == Image.Image:
		im = img_path
		
	else:
		im = Image.open(img_path).convert('LA').resize((200,200))

	if filled == True:
		imcv = cv2.imread(img_path,-1)[:,:,3]
		filled_im = fill_im(imcv)
		# cv2_im = cv2.cvtColor(cv_im,cv2.COLOR_BGR2RGB)
		im = Image.fromarray(np.uint8(filled_im * 255),'L')
		data = np.array(im)
		# dat = data[:,:,1]
		dat = data
	else:
		if actual == True:
			data = np.array(im)
			dat_ = data[:,:,0]
			dat = util.invert(dat_)
		else:
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
	if len(data_total.shape) > 2:
		if actual == True:
			img_data = util.invert(data_total[:,:,0])
		else:
			img_data = data_total[:,:,1]
	else:
		img_data = data_total
	# imdat = Image.fromarray(img_data)

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
		part_1 = im1
		part_2 = im2
	else:
		axis = 2
		part_1 = im3
		part_2 = im4
		
	for im_m in [im1,im2,im3,im4]:
		if im_m.mode != 'RGB':
			im_m = im_m.convert('RGB')

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

	part_5 = Image.fromarray(dat5).convert('RGB')
	part_6 = Image.fromarray(dat6).convert('RGB')
	part_7 = Image.fromarray(dat7).convert('RGB')
	part_8 = Image.fromarray(dat8).convert('RGB')
	part_9 = Image.fromarray(dat9).convert('RGB')
	part_10 = Image.fromarray(dat10).convert('RGB')

	part_tup = (part_1,part_2,part_5,part_6,part_7,part_8,part_9,part_10)
	
	if output_path == False:
		return part_tup

	else:
		imname = img_path.split('/')[-1]
		name,ext = imname.split('.')
		outdir = final_path
		output_name_base = outdir + '/' + name
		for ct in range(len(part_tup)):
			part = part_tup[ct]
			part.save(output_name_base + "part_%s"%ct + ext)

		return part_tup

########### DOMINANT COLOR FINDING ################

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

def dominant_colors(impath_texture,mask):
    im = Image.open(impath_texture).resize(mask.shape)
    imcv = pil_to_cv(im)
    res = cv2.bitwise_and(imcv,imcv,mask = mask)
    img_r = res.reshape((res.shape[0] * res.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=5) #cluster number
    clt.fit(img_r)

    colors = clt.cluster_centers_

    bad = []
    for i in range(len(colors)):
        color = colors[i]
        if color[0] > 245 and color[1] > 245 and color[2] > 245:
            bad.append(i)
        elif color[0] < 35 and color[1] < 35 and color[2] < 35:
            bad.append(i)
    
    hist = find_histogram(clt)
    colors = clt.cluster_centers_

    hist_new = []
    colors_new = []

    hist_sum = 0
    for i in range(len(hist)):
        if i not in bad:
            hist_new.append(hist[i])
            hist_sum += hist[i]
            colors_new.append(colors[i])
    hist_new = [h/hist_sum for h in hist_new]
    bar = plot_colors2(hist_new, colors_new)

    return hist_new,colors_new,bar