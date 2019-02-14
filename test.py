from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import sklearn.datasets
import time
from PIL import ImageFile
from PIL import Image
from video_folder_test import VideoFolder
from torch.utils.data import DataLoader
from Dynamic_Model import MDGAN_S1_G, MDGAN_S2_G, SubsetRandomSampler
ImageFile.LOAD_TRUNCATED_IMAGES = True
cudnn.benchmark = True


parser = argparse.ArgumentParser()
parser.add_argument('--netG_S2', 
	default='./netG_S2_067.pth', 
	help='path to netG Stage 2')
parser.add_argument('--netG_S1', 
	default='./netG_S1_030.pth', 
	help='path to netG Stage 1')
parser.add_argument('--cuda', action='store_true', default=True, 
	help='enables cuda')
parser.add_argument('--outf', default = './results', 
	help='output folder')
parser.add_argument('--testf', default = '../../sky_timelapse/sky_test', 
	help='test data folder')
opt = parser.parse_args()


localtime = time.asctime( time.localtime(time.time()) )
print('\n start new program! ')
print(localtime)

cuda = opt.cuda
test_folder = opt.testf
output_path = opt.outf
if not os.path.exists(output_path):
	os.mkdir(output_path)

imageSize = 128
netG_S1 = MDGAN_S1_G(32)
netG_S1_path = opt.netG_S1
netG_S2 = MDGAN_S2_G(32)
netG_S2_path = opt.netG_S2


# Load models
netG_S1.load_state_dict(torch.load(opt.netG_S1))
netG_S2.load_state_dict(torch.load(opt.netG_S2))

# -------------set inputs---------------------------------
testset = VideoFolder(root=test_folder,
					nframes = 32,
					transform=transforms.Compose([
								transforms.Resize( (imageSize, imageSize) ),
								transforms.ToTensor(),
								transforms.Normalize((0.5, 0.5, 0.5),
									(0.5, 0.5, 0.5)),
									]))

print('testset size: ' + str (len(testset) ) )


valid_loader = DataLoader(testset,
					batch_size=48,
					num_workers=1,
					shuffle=True,
					drop_last = True, 
					pin_memory=False)						 
valid_iter = iter(valid_loader)
val_gt, _ = valid_iter.next()
print('validation video loaded.')

if cuda:
	netG_S1.cuda()
	netG_S2.cuda()
	val_gt = val_gt.cuda()
	
netG_S1.train()
netG_S2.train()

val_video = val_gt[:,:,0,:,:]
val_video = val_video.unsqueeze(2).repeat(1,1,val_gt.size(2),1,1 )	
val_fake_s1 =  netG_S1(Variable(val_video))
val_fake_s2 = netG_S2(val_fake_s1) # size: batchsize * 3 * 32 * 64 *64
val_fake_s1 = val_fake_s1.data.permute(2,0,1,3,4) 
# permute to 32 * batchsize * 3 *64 *64
val_fake_s2 = val_fake_s2.data.permute(2,0,1,3,4)

# save fake samples of stage 1
for t in range(val_fake_s1.size(0)):
	vutils.save_image(val_fake_s1[t], 
		'%s/samples_s1_frame_%03d.png'
		% (opt.outf, t),normalize=True, 
		nrow = 8)				 
# save fake samples of stage 2
for t in range(val_fake_s2.size(0)):
	vutils.save_image(val_fake_s2[t],
		'%s/samples_s2_frame_%03d.png'
		% (opt.outf, t),normalize=True, 
		nrow = 8)
# save real samples
for t in range(val_gt.permute(2,0,1,3,4).size(0)):
	vutils.save_image(val_gt.permute(2,0,1,3,4)[t],
		'%s/samples_real_frame_%03d.png'
		% (opt.outf, t),normalize=True, 
		nrow = 8)	 

def generate_video(model='s1', outf= opt.outf):
	img_path = os.path.join(outf, 'samples_' + model +	'_frame_%03d.png')
	mp4_path = os.path.join(outf, model+ '_video.mp4')
	cmd = ('ffmpeg -loglevel warning -framerate 25 -i ' + img_path + 
		' -qscale:v 2 -y ' + mp4_path )
	print(cmd)
	os.system(cmd)
generate_video('s1')
generate_video('s2')
generate_video('real')





