import torch
import torch.nn as nn
#import torch
import numpy as np
import os
import cv2 as cv
import torchvision
import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.utils as vutils
import glob
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import re, PIL, math
from PIL import Image
import random, time
from torch.nn import init
from functools import partial
import tqdm

def get_augmented_images_salient(img, rotation = 0, diff = 1):
	res = []
	res.append(img)
	res.append(img.transpose(PIL.Image.FLIP_TOP_BOTTOM))
	res.append(img.transpose(PIL.Image.FLIP_LEFT_RIGHT))
	if rotation:
	    for i in range(-rotation, rotation + 1, diff):
	        #print("Inside This")
	        res.append(img.rotate(i, resample=Image.BILINEAR, expand = False))
	# else:
	#     res.append(img.rotate(0, resample=Image.BILINEAR, expand = False))
	return res

def gen_ctr(IMG, kernel_size=5):
	kernel = np.ones((kernel_size, kernel_size))
	C = cv.dilate(IMG, kernel) - cv.erode(IMG, kernel)
	return C

def get_mae(model, dataloader, cuda, im_size):
	model.eval()
	# print(model.training, flush = True)
	iou_l = 0.0
	mae = 0.0
	j = nn.Sigmoid()
	count = 0
	pred_list = []
	gt_list = []
	transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((im_size, im_size)), transforms.ToTensor()])
	for x, z, _ in dataloader:
		count += 1
		_, _, h, w = x.size()
		x = transform(x.squeeze(0)).unsqueeze(0)
		# print(x.size(), flush = True)
		prediction = model(x.to(cuda))[-1]
		pred = j(prediction[-1])
		up = nn.Upsample(size=(h, w), mode='bilinear',align_corners=False)
		pred = up(pred)
		# print(pred.size())
		for i in range(len(pred)):
			pred_list.append((pred[i, :, :, :] / torch.max(pred[i, :, :, :])).detach().squeeze().cpu().numpy())
			gt_list.append(z[i, :, :, :].detach().squeeze().cpu().numpy())
		mae += MAE(pred.cpu(), z.cpu())
	print("MAE = %.4f" %(mae / count), flush = True)
	return pred_list, gt_list, mae / count

def get_mae_depth(model, dataloader, cuda, im_size):
	# model.eval()
	iou_l = 0.0
	mae = 0.0
	j = nn.Sigmoid()
	count = 0
	pred_list = []
	gt_list = []
	transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((im_size, im_size)), transforms.ToTensor()])
	for x, z, d in dataloader:
		count += 1
		_, _, h, w = x.size()
		if type(model) is list:
			x = transform(x.squeeze(0)).unsqueeze(0)
			d = transform(d.squeeze(0)).unsqueeze(0)
			depth_features, _ = model[1](d.to(cuda))[:2]
			prediction = model[0](x.to(cuda), depth_features)[-1]
		else:
			x = transform(x.squeeze(0)).unsqueeze(0)
			d = transform(d.squeeze(0)).unsqueeze(0)
			prediction = model(x.to(cuda), d.to(cuda))[-1]
		# prediction = model(x.to(cuda), d.to(cuda))[-1]
		pred = j(prediction[-1])
		up = nn.Upsample(size=(h, w), mode='bilinear',align_corners=False)
		pred = up(pred)
		# print(pred.size())
		for i in range(len(pred)):
			pred_list.append((pred[i, :, :, :] / torch.max(pred[i, :, :, :])).detach().squeeze().cpu().numpy())
			gt_list.append(z[i, :, :, :].detach().squeeze().cpu().numpy())
		mae += MAE(pred.cpu(), z.cpu())
	print("MAE = %.4f" %(mae / count), flush = True)
	return pred_list, gt_list, mae / count

def compute_pre_rec(gt,mask,mybins=np.arange(0,256)):

	if(len(gt.shape)<2 or len(mask.shape)<2):
		print("ERROR: gt or mask is not matrix!")
		exit()
	if(len(gt.shape)>2): # convert to one channel
		gt = gt[:,:,0]
	if(len(mask.shape)>2): # convert to one channel
		mask = mask[:,:,0]
	if(gt.shape!=mask.shape):
		print("ERROR: The shapes of gt and mask are different!")
		exit()

	gtNum = gt[gt>128].size # pixel number of ground truth foreground regions
	pp = mask[gt>128] # mask predicted pixel values in the ground truth foreground region
	nn = mask[gt<=128] # mask predicted pixel values in the ground truth bacground region

	pp_hist,pp_edges = np.histogram(pp,bins=mybins) #count pixel numbers with values in each interval [0,1),[1,2),...,[mybins[i],mybins[i+1]),...,[254,255)
	nn_hist,nn_edges = np.histogram(nn,bins=mybins)

	pp_hist_flip = np.flipud(pp_hist) # reverse the histogram to the following order: (255,254],...,(mybins[i+1],mybins[i]],...,(2,1],(1,0]
	nn_hist_flip = np.flipud(nn_hist)

	pp_hist_flip_cum = np.cumsum(pp_hist_flip) # accumulate the pixel number in intervals: (255,254],(255,253],...,(255,mybins[i]],...,(255,0]
	nn_hist_flip_cum = np.cumsum(nn_hist_flip)

	precision = pp_hist_flip_cum/(pp_hist_flip_cum + nn_hist_flip_cum+1e-8) #TP/(TP+FP)
	recall = pp_hist_flip_cum/(gtNum+1e-8) #TP/(TP+FN)

	precision[np.isnan(precision)]= 0.0
	recall[np.isnan(recall)] = 0.0

	return np.reshape(precision,(len(precision))),np.reshape(recall,(len(recall)))


def compute_PRE_REC_FM_of_methods(gt_name_list,pred_name_list,rs_dir_lists=1,beta=0.3):
#input 'gt_name_list': ground truth name list
#input 'rs_dir_lists': to-be-evaluated mask directories (not the file names, just folder names)
#output precision 'PRE': numpy array with shape of (num_rs_dir, 256)
#       recall    'REC': numpy array with shape of (num_rs_dir, 256)
#       F-measure (beta) 'FM': numpy array with shape of (num_rs_dir, 256)

	mybins = np.arange(0,256) # different thresholds to achieve binarized masks for pre, rec, Fm measures

	num_gt = len(gt_name_list) # number of ground truth files
	num_rs_dir = 1 # number of method folders
	if(num_gt==0):
		#print("ERROR: The ground truth directory is empty!")
		exit()

	PRE = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # PRE: with shape of (num_gt, num_rs_dir, 256)
	REC = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # REC: the same shape with PRE
	# FM = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # Fm: the same shape with PRE
	gt2rs = np.zeros((num_gt,num_rs_dir)) # indicate if the mask of methods is correctly computed

	for i in range(0,num_gt):
		print('>>Processed %d/%d'%(i+1,num_gt),end='\r')
		gt = gt_name_list[i]# read ground truth
		gt = gt*255.0 # convert gt to [0,255]
		#gt_name = gt_name_list[i].split('/')[-1] # get the file name of the ground truth "xxx.png"

		for j in range(0,num_rs_dir):
			pre, rec, f = np.zeros(len(mybins)), np.zeros(len(mybins)), np.zeros(len(mybins)) # pre, rec, f or one mask w.r.t different thresholds
			try:
				rs = pred_name_list[i] # read the corresponding mask from each method
				rs = rs*255.0 # convert rs to [0,255]
			except IOError:
				#print('ERROR: Couldn\'t find the following file:',rs_dir_lists[j]+gt_name)
				continue
			try:
				pre, rec = compute_pre_rec(gt,rs,mybins=np.arange(0,256))
			except IOError:
				#print('ERROR: Fails in compute_mae!')
				continue

			PRE[i,j,:] = pre
			REC[i,j,:] = rec
			gt2rs[i,j] = 1.0
	print('\n')
	gt2rs = np.sum(gt2rs,0) # num_rs_dir
	gt2rs = np.repeat(gt2rs[:, np.newaxis], 255, axis=1) #num_rs_dirx255

	PRE = np.sum(PRE,0)/(gt2rs+1e-8) # num_rs_dirx255, average PRE over the whole dataset at every threshold
	REC = np.sum(REC,0)/(gt2rs+1e-8) # num_rs_dirx255
	FM = (1+beta)*PRE*REC/(beta*PRE+REC+1e-8) # num_rs_dirx255

	return PRE, REC, FM, gt2rs

def validation(model, dataloader, cuda, im_size, use_depth = False):
	with torch.no_grad():
		if use_depth:
			pred_list, gt_list, mae = get_mae_depth(model, dataloader, cuda, im_size)
		else:
			pred_list, gt_list, mae = get_mae(model, dataloader, cuda, im_size)
		# print('MAE = %.4f')
		PRE, REC, FM, gt2rs_fm = compute_PRE_REC_FM_of_methods(gt_list,pred_list,1, beta=0.3)
		for i in range(0,FM.shape[0]):
			print(">>", "My Method",":", "num_rs/num_gt-> %d/%d,"%(int(gt2rs_fm[i][0]),len(gt_list)), "maxF->%.3f, "%(np.max(FM,1)[i]), "meanF->%.3f, "%(np.mean(FM,1)[i]))
		#print('\n')
	# return np.mean(FM,1)[0], mae
	return mae

class USODAugmentedLoader(Dataset):
	def __init__(self, augment_data=False, transform=None, im_size=256):

		self.inp_path = './AugmentedUSOD/RGB'
		self.out_path = './AugmentedUSOD/GT'
		self.contour_path = './AugmentedUSOD/Boundary'
		self.depth_path = './AugmentedUSOD/Depth'

		self.transform = transform
		self.augment_data = augment_data

		if im_size != 384:
			self.im_size = im_size
		else:
			self.im_size = None

		if self.transform:
			self.transform_aug = transforms.Compose([
				transforms.ToPILImage(),
				transforms.GaussianBlur(5),
				transforms.ToTensor()
		])

		self.inp_files = sorted(glob.glob(self.inp_path + '/*'))#[:100]
		self.out_files = sorted(glob.glob(self.out_path + '/*'))#[:100]
		self.depth_files = sorted(glob.glob(self.depth_path + '/*'))#[:100]
		self.contour_files = sorted(glob.glob(self.contour_path + '/*'))#[:100]

	def __getitem__(self, idx):
		inp_img = cv.imread(self.inp_files[idx])
		inp_img = cv.cvtColor(inp_img, cv.COLOR_BGR2RGB)
		inp_img = inp_img.astype('float32')

		# mask_img = cv.imread(self.out_files[idx], 0)
		mask_img = cv.imread('./AugmentedUSOD/GT/' + self.inp_files[idx].split('/')[-1], 0)
		# print(mask_img.shape)
		mask_img = mask_img.astype('float32')
		mask_img /= np.max(mask_img)

		# depth_img = cv.imread(self.depth_files[idx], 0)
		depth_img = cv.imread('./AugmentedUSOD/Depth/' + self.inp_files[idx].split('/')[-1], 0)
		depth_img = depth_img.astype('float32')
		depth_img /= np.max(depth_img)

		# contour_img = cv.imread(self.contour_files[idx], 0)
		contour_img = cv.imread('./AugmentedUSOD/Boundary/' + self.inp_files[idx].split('/')[-1], 0)
		contour_img = contour_img.astype('float32')
		contour_img /= np.max(contour_img)

		if self.augment_data:
			inp_img = random_brightness(inp_img)

		if self.im_size is not None:
			im_size = self.im_size
			inp_img = cv.resize(inp_img, (im_size, im_size), interpolation = cv.INTER_AREA)
			mask_img = cv.resize(mask_img, (im_size, im_size), interpolation = cv.INTER_AREA)
			contour_img = cv.resize(contour_img, (im_size, im_size), interpolation = cv.INTER_AREA)
			depth_img = cv.resize(depth_img, (im_size, im_size), interpolation = cv.INTER_AREA)


		inp_img /= np.max(inp_img)
		inp_img = np.transpose(inp_img, axes=(2, 0, 1))
		inp_img = torch.from_numpy(inp_img).float()

		mask_img = np.expand_dims(mask_img, axis=0)

		depth_img = np.expand_dims(depth_img, axis=0)

		contour_img = np.expand_dims(contour_img, axis=0)

		if self.transform:
			aug_img = self.transform_aug(inp_img)
			return inp_img, aug_img, torch.from_numpy(mask_img).float(), torch.from_numpy(contour_img).float(), torch.from_numpy(depth_img).float()

		return inp_img, torch.from_numpy(mask_img).float(), torch.from_numpy(contour_img).float(), torch.from_numpy(depth_img).float()

	def __len__(self):
		return len(self.inp_files)

class RGBDSODLoader(Dataset):
	def __init__(self, augment_data=False, transform=None, im_size=256):

		self.inp_path = './train_data/RGB'
		self.out_path = './train_data/GT'
		self.depth_path = './train_data/depth'

		self.transform = transform
		self.augment_data = augment_data

		if im_size != 384:
			self.im_size = im_size
		else:
			self.im_size = 384

		if self.transform:
			self.transform_aug = transforms.Compose([
				transforms.ToPILImage(),
				transforms.GaussianBlur(5),
				transforms.ToTensor()
		])

		self.inp_files = sorted(glob.glob(self.inp_path + '/*'))
		self.out_files = sorted(glob.glob(self.out_path + '/*'))
		self.depth_files = sorted(glob.glob(self.depth_path + '/*'))

	def __getitem__(self, idx):
		inp_img = cv.imread(self.inp_files[idx])
		inp_img = cv.cvtColor(inp_img, cv.COLOR_BGR2RGB)
		inp_img = inp_img.astype('float32')

		mask_img = cv.imread('./train_data/GT/' + self.inp_files[idx].split('/')[-1].split('.jpg')[0] + '.png', 0)
		mask_img = mask_img.astype('float32')
		mask_img /= np.max(mask_img)

		depth_img = cv.imread('./train_data/depth/' + self.inp_files[idx].split('/')[-1].split('.jpg')[0] + '.png', 0)
		depth_img = depth_img.astype('float32')
		depth_img /= np.max(depth_img)

		contour_img = gen_ctr(mask_img)

		if self.augment_data:
			inp_img = random_brightness(inp_img)

		if self.im_size is not None:
			im_size = self.im_size
			inp_img = cv.resize(inp_img, (im_size, im_size), interpolation = cv.INTER_AREA)
			mask_img = cv.resize(mask_img, (im_size, im_size), interpolation = cv.INTER_AREA)
			contour_img = cv.resize(contour_img, (im_size, im_size), interpolation = cv.INTER_AREA)
			depth_img = cv.resize(depth_img, (im_size, im_size), interpolation = cv.INTER_AREA)


		inp_img /= np.max(inp_img)
		inp_img = np.transpose(inp_img, axes=(2, 0, 1))
		inp_img = torch.from_numpy(inp_img).float()

		mask_img = np.expand_dims(mask_img, axis=0)

		depth_img = np.expand_dims(depth_img, axis=0)

		contour_img = np.expand_dims(contour_img, axis=0)

		if self.transform:
			aug_img = self.transform_aug(inp_img)
			return inp_img, aug_img, torch.from_numpy(mask_img).float(), torch.from_numpy(contour_img).float(), torch.from_numpy(depth_img).float()

		return inp_img, torch.from_numpy(mask_img).float(), torch.from_numpy(contour_img).float(), torch.from_numpy(depth_img).float()

	def __len__(self):
		return len(self.inp_files)

class AugmentedRGBDSODLoader(Dataset):
	"""docstring for AugmentedDataset"""
	def __init__(self, transform = False, im_size = 384):
		super(AugmentedRGBDSODLoader, self).__init__()
		self.dataset = RGBDSODLoader(transform = transform, im_size = im_size)
		self.transf = transforms.ToTensor()
		self.transfx = transforms.ToTensor()
		self.create_data = []
		self.getitem()

	def getitem(self):
		for i in range(len(self.dataset)):
			image = transforms.ToPILImage()(self.dataset[i][0])
			saliency = transforms.ToPILImage()(self.dataset[i][1])
			contour = transforms.ToPILImage()(self.dataset[i][2])
			depth = transforms.ToPILImage()(self.dataset[i][3])
			for i, j, k, d in zip(
				get_augmented_images_salient(image, 90, 45), 
				get_augmented_images_salient(saliency, 90, 45), 
				get_augmented_images_salient(contour, 90, 45),
				get_augmented_images_salient(depth, 90, 45)
			):
				self.create_data.append([self.transfx(i), self.transf(j), self.transf(k), self.transf(d)])

	def __getitem__(self, i):
		return self.create_data[i]

	def __len__(self):
		return len(self.create_data)

class RGBDSODTestLoader(Dataset):
	def __init__(self, with_name = False):
		
		self.inp_path = './test_data/NJU2K/RGB'
		self.out_path = './test_data/NJU2K/GT'

		self.inp_files = sorted(glob.glob(self.inp_path + '/*'))
		self.out_files = sorted(glob.glob(self.out_path + '/*'))

		self.depth_path = './test_data/NJU2K/depth'
		self.depth_files = sorted(glob.glob(self.depth_path + '/*'))

		self.with_name = with_name

	def __getitem__(self, idx):
		inp_img = cv.imread(self.inp_files[idx])
		inp_img = cv.cvtColor(inp_img, cv.COLOR_BGR2RGB)
		inp_img = inp_img.astype('float32')

		mask_img = cv.imread('./test_data/NJU2K/GT/' + self.inp_files[idx].split('/')[-1].split('.jpg')[0] + '.png', 0)
		mask_img = mask_img.astype('float32')
		mask_img /= np.max(mask_img)

		inp_img /= np.max(inp_img)
		inp_img = np.transpose(inp_img, axes=(2, 0, 1))
		inp_img = torch.from_numpy(inp_img).float()

		mask_img = np.expand_dims(mask_img, axis=0)

		depth_img = cv.imread('./test_data/NJU2K/depth/'+ self.inp_files[idx].split('/')[-1].split('.jpg')[0] + '.png', 0)
		depth_img = depth_img.astype('float32')
		depth_img /= np.max(depth_img)
		depth_img = np.expand_dims(depth_img, axis=0)

		if self.with_name:
			return inp_img, torch.from_numpy(mask_img).float(), torch.from_numpy(depth_img).float(), self.inp_files[idx]
		return inp_img, torch.from_numpy(mask_img).float(), torch.from_numpy(depth_img).float()

	def __len__(self):
		return len(self.inp_files)

class RGBDSODTestLoaderAll(Dataset):
	def __init__(self, path = './test_data/NJU2K',  with_name = False):
		
		self.inp_path = path + '/RGB'
		self.out_path = path + '/GT'

		self.inp_files = sorted(glob.glob(self.inp_path + '/*'))
		self.out_files = sorted(glob.glob(self.out_path + '/*'))

		self.depth_path = path + '/depth'
		self.depth_files = sorted(glob.glob(self.depth_path + '/*'))

		self.with_name = with_name

		self.path = path

	def __getitem__(self, idx):
		inp_img = cv.imread(self.inp_files[idx])
		inp_img = cv.cvtColor(inp_img, cv.COLOR_BGR2RGB)
		inp_img = inp_img.astype('float32')

		mask_img = cv.imread(self.path + '/GT/' + self.inp_files[idx].split('/')[-1].split('.jpg')[0] + '.png', 0)
		mask_img = mask_img.astype('float32')
		mask_img /= np.max(mask_img)

		inp_img /= np.max(inp_img)
		inp_img = np.transpose(inp_img, axes=(2, 0, 1))
		inp_img = torch.from_numpy(inp_img).float()

		mask_img = np.expand_dims(mask_img, axis=0)

		depth_img = cv.imread(self.path + '/depth/'+ self.inp_files[idx].split('/')[-1].split('.jpg')[0] + '.png', 0)
		depth_img = depth_img.astype('float32')
		depth_img /= np.max(depth_img)
		depth_img = np.expand_dims(depth_img, axis=0)

		if self.with_name:
			return inp_img, torch.from_numpy(mask_img).float(), torch.from_numpy(depth_img).float(), self.inp_files[idx]
		return inp_img, torch.from_numpy(mask_img).float(), torch.from_numpy(depth_img).float()

	def __len__(self):
		return len(self.inp_files)

class USODTestLoader(Dataset):
	def __init__(self, with_name = False):
		
		self.inp_path = './USOD10k/TE/RGB/RGB'
		self.out_path = './USOD10k/TE/GT/GT/GT'

		self.inp_files = sorted(glob.glob(self.inp_path + '/*'))
		self.out_files = sorted(glob.glob(self.out_path + '/*'))

		self.depth_path = './USOD10k/TE/depth/depth'
		self.depth_files = sorted(glob.glob(self.depth_path + '/*'))

		self.with_name = with_name

	def __getitem__(self, idx):
		inp_img = cv.imread(self.inp_files[idx])
		inp_img = cv.cvtColor(inp_img, cv.COLOR_BGR2RGB)
		inp_img = inp_img.astype('float32')

		mask_img = cv.imread(self.out_files[idx], 0)
		mask_img = mask_img.astype('float32')
		mask_img /= np.max(mask_img)

		inp_img /= np.max(inp_img)
		inp_img = np.transpose(inp_img, axes=(2, 0, 1))
		inp_img = torch.from_numpy(inp_img).float()

		mask_img = np.expand_dims(mask_img, axis=0)

		depth_img = cv.imread(self.depth_files[idx], 0)
		depth_img = depth_img.astype('float32')
		depth_img /= np.max(depth_img)
		depth_img = np.expand_dims(depth_img, axis=0)

		if self.with_name:
			return inp_img, torch.from_numpy(mask_img).float(), torch.from_numpy(depth_img).float(), self.inp_files[idx]
		return inp_img, torch.from_numpy(mask_img).float(), torch.from_numpy(depth_img).float()

	def __len__(self):
		return len(self.inp_files)
