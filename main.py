import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets as dsets
from torchvision import transforms as trans
from torch.autograd import Variable as V
import numpy as np
import torchvision.models as models
import pdb
from logger import Logger
import argparse
import shutil
import os.path
from plot_confusion_matrix import plot_conf_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
# ------------- (Start) Hyper Parameters ---------------
bs = 60 # Batch Size
learning_rate = 1e-3
wd = 0 # weight_decay
itr = 40
cuda = True
is_best = True
class_names = ['0', '1']

# ------------- (End) Hyper Parameters ---------------

torch.manual_seed(0)
if torch.cuda.is_available() and cuda:
	torch.cuda.manual_seed_all(0)
	FloatType = torch.cuda.FloatTensor
	LongType = torch.cuda.LongTensor
else:
	FloatType = torch.FloatTensor
	LongType = torch.LongTensor

# Define transformation
train_trans = trans.Compose([
		trans.ToTensor()])
val_trans = trans.Compose([
		trans.ToTensor()])

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Loading dataset
train_data = dsets.ImageFolder(root = '/media/dani/00E0B705E0B6FFC8/dataset_pre_kaggle/train/', transform = train_trans)
val_data =   dsets.ImageFolder(root = '/media/dani/00E0B705E0B6FFC8/dataset_pre_kaggle/val/', transform = val_trans)
train_loader = torch.utils.data.DataLoader(train_data, batch_size = bs, shuffle = True, num_workers = 3)
test_loader = torch.utils.data.DataLoader(val_data, batch_size = bs, shuffle = False, num_workers = 3)


def to_np(x):
    return x.data.cpu().numpy()

def save_checkpoint(state, filename='checkpoint.pth.tar'):
	torch.save(state, filename)
	# if is_best:
	# 	shutil.copyfile(filename, 'model_best.pth.tar')

def weights_init(m):
	if isinstance(m, nn.Conv2d):
		torch.nn.init.kaiming_normal(m.weight.data)
	elif isinstance(m, nn.Linear):
		torch.nn.init.kaiming_normal(m.weight.data)
		m.bias.data.normal_(mean=0,std=1e-2)
	elif isinstance(m, nn.BatchNorm2d):
		m.weight.data.uniform_()
		m.bias.data.zero_()

# Model and Optimizer definition
logger = Logger('./logs')
model = models.resnet34(pretrained = True)
in_features = model.fc.in_features
num_class = len(train_loader.dataset.classes)
model.avgpool = nn.AvgPool2d(10,10)
model.fc = nn.Linear(in_features, num_class)

model = torch.nn.DataParallel(model, device_ids=[0])

if cuda:
	model = model.cuda()

model.module.fc.apply(weights_init)

optimizer = optim.Adam(params = model.parameters(), lr = learning_rate, weight_decay=0)
criterion = torch.nn.CrossEntropyLoss(FloatType([9,25]))
global args
args = parser.parse_args()
if args.resume:
	if os.path.isfile(args.resume):
		print("=> loading checkpoint '{}'".format(args.resume))
		checkpoint = torch.load(args.resume)
		args.start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print("=> loaded checkpoint '{}' (epoch {})"
			  .format(args.resume, checkpoint['epoch']))
	else:
		print("=> no checkpoint found at '{}'".format(args.resume))


def train_model(model, optimizer, train_loader, criterion, epoch, vis_step = 2):
	# pdb.set_trace()
	model.train(mode = True)
	num_hit = 0
	total = len(train_loader.dataset)
	num_batch = np.ceil(total/bs)
	# Training Phase on train dataset
	for batch_idx, (image, labels) in enumerate(train_loader):
		optimizer.zero_grad()
		pdb.set_trace()
		image, labels = V(image.type(FloatType)), V(labels.type(LongType))
		# pdb.set_trace()
		output = model(image)
		loss = criterion(output, labels)
		if batch_idx % vis_step == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(image),
					len(train_loader.dataset),
					100. * batch_idx / len(train_loader),
					loss.data[0]))
		loss.backward()
		optimizer.step()
	# Validation Phase on train dataset
	model.eval()
	for batch_idx, (image, labels) in enumerate(train_loader):
		image, labels = V(image.type(FloatType), volatile=True), V(labels.type(LongType), volatile=True)
		output = model(image)
		_ , pred_label = output.data.max(dim=1)
		num_hit += (pred_label == labels.data).sum()
	train_accuracy = (num_hit / total)
	print("Epoch: {}, Training Accuracy: {:.2f}%".format(epoch, 100. * train_accuracy))
	save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict(),
			})



	info = {
			'loss_train':train_accuracy,
		}

	for tag, value in info.items():
		logger.scalar_summary(tag, value, epoch)
	for tag, value in model.named_parameters():
		tag = tag.replace('.', '/')
		logger.histo_summary(tag, to_np(value), epoch)
		logger.histo_summary(tag+'/grad', to_np(value.grad), epoch)
	return 100. * train_accuracy

def eval_model(model, test_loader, epoch):
	model.train(mode = False)
	num_hit = 0
	total = len(test_loader.dataset)
	for batch_idx, (image, labels) in enumerate(test_loader):
		image, labels = V(image.type(FloatType), volatile=True), V(labels.type(LongType), volatile=True)
		output = model(image)
		# pdb.set_trace()
		_ , pred_label = output.data.max(dim=1)
		num_hit += (pred_label == labels.data).sum()
		true.extend(labels.data.cpu().numpy().tolist())
		pred.extend(pred_label.cpu().numpy().tolist())
	# pdb.set_trace()
	test_accuracy = (num_hit / total)
	print("Epoch: {}, Testing Accuracy: {:.2f}%".format(epoch, 100. * test_accuracy))
	info = {
			'loss_test':test_accuracy,
		}

	for tag, value in info.items():
		logger.scalar_summary(tag, value, epoch)
	# pdb.set_trace()
	# is_best = test_accuracy > best_accuracy
	# best_accuracy = max(best_accuracy, test_accuracy)
	return 100. * test_accuracy

train_acc = []
test_acc = []
true = []
pred = []
best_accuracy = 0
for epoch in range(itr):
	tr_acc = train_model(model, optimizer, train_loader, criterion, epoch,200)
	ts_acc = eval_model(model, test_loader, epoch)
	# train_acc.append(tr_acc)
	test_acc.append(ts_acc)
	pdb.set_trace()
	plot_conf_matrix(true, pred, class_names)
	print("cohen kapa score:",cohen_kappa_score(true, pred))
	target_names = ['class 0', 'class 1']
	print(classification_report(true, pred, target_names=target_names))
