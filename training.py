import argparse
from utils import *
from torch.nn import functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
sys.path.append("../checkpoints")
from USODFuseNet import *
import tqdm

def re_Dice_Loss(inputs, targets, cuda=False, balance=1.1):
	n, c, h, w = inputs.size()
	smooth=1
	inputs = torch.sigmoid(inputs)

	input_flat=inputs.view(-1)
	target_flat=targets.view(-1)

	intersecion=input_flat*target_flat
	unionsection=input_flat.pow(2).sum()+target_flat.pow(2).sum()+smooth
	loss=unionsection/(2*intersecion.sum()+smooth)
	loss=loss.mean()

	return loss

def _weighted_cross_entropy_loss(preds, edges, device, weight = 10):
	mask = (edges == 1.0).float()
	b, c, h, w = edges.shape
	num_pos = torch.sum(mask, dim=[1, 2, 3]).float()
	num_neg = c * h * w - num_pos
	weight = torch.zeros_like(edges)
	nx1 = num_neg / (num_pos + num_neg)
	nx2 = num_pos / (num_pos + num_neg)
	weight = torch.cat([torch.where(i == 1.0, j, k) for i, j, k in zip(edges, nx1, nx2)], dim = 0).unsqueeze(1)
	losses = F.binary_cross_entropy_with_logits(preds.float(),
				edges.float(),
				weight=weight,
				reduction='none')
	loss = torch.sum(losses) / b
	return loss

def structure_loss_contour(pred,target):

	bce_out = _weighted_cross_entropy_loss(pred,target,None)
	iou_out = re_Dice_Loss(pred, target)
	pred = torch.sigmoid(pred)

	loss = 0.001 * bce_out + iou_out

	return loss

def structure_loss_salient_fb(pred, mask, foreground, weight=None):
	if foreground:
		assert weight is not None
		weit  = 1+5*weight
		wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce=False)
		wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

		pred  = torch.sigmoid(pred)
		inter = ((pred*mask)*weit).sum(dim=(2,3))
		union = ((pred+mask)*weit).sum(dim=(2,3))
		wiou  = 1-(inter+1)/(union-inter+1)

		mae = F.l1_loss(pred, mask, reduce=False)
		wmae = (mae*weit).sum(dim=(2,3))/weit.sum(dim=(2,3))
		return (wbce + wiou + wmae).mean()
	else:
		mask = 1.-mask
		weit  = 1+5*mask
		wbce = F.binary_cross_entropy_with_logits(-1*pred, mask, reduce=False)
		wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

		pred  = 1.-torch.sigmoid(pred)
		inter = ((pred*mask)*weit).sum(dim=(2,3))
		union = ((pred+mask)*weit).sum(dim=(2,3))
		wiou  = 1-(inter+1)/(union-inter+1)

		mae = F.l1_loss(pred, mask, reduce=False)
		wmae = (mae*weit).sum(dim=(2,3))/weit.sum(dim=(2,3))
		return (wbce + wiou + wmae).mean()

def init_weights(net, init_type='normal', gain=0.02):
	def init_func(m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal' and classname.find('Conv1d') != -1:
				n = m.kernel_size[0] * m.in_channels
				init.normal_(m.weight.data, 0.0, math.sqrt(2.0 / n))
			elif init_type == 'normal' and classname.find('Conv') != -1:
				n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
				init.normal_(m.weight.data, 0.0, math.sqrt(2.0 / n))
			elif init_type == 'normal' and classname.find('Linear') != -1:
				init.normal_(m.weight.data, 0.0, 1.0)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=gain)
			else:
				raise NotImplementedError('initialization method [%s] is not implemented for [%s]' % (init_type, classname))
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1 or classname.find('BatchNorm1d') != -1:
			init.normal_(m.weight.data, 1.0, gain)
			init.constant_(m.bias.data, 0.0)

	print('initialize network with %s' % init_type)
	net.apply(init_func)

def training_deepx(

		epochs, 
		sv_threshold, 
		model, 
		dataloader, 
		test_loader,
		cuda, 
		optimizer, 
		scheduler = None, 
		f_name = 'model.pt', 
		loss_threshold = None,
		salient_loss_weight = 1.0,
		im_size = None
	):


	loss_before = 0.0
	scaler = torch.cuda.amp.GradScaler(enabled = True)
	old_mae = 1.0
	for epoch in range(epochs):
		if use_depth:
			model[0].train()
			model[1].train()
		else:
			model.train()
		loss_end = 0.0
		count = 0
		for i, data in enumerate(tqdm.tqdm(dataloader)):
			images = data[0].to(device = cuda)
			saliency = data[1].to(device = cuda)
			contour = data[2].to(device = cuda)
			depth = data[3].to(device = cuda)
			with torch.autocast(device_type = 'cuda', dtype = torch.float16, enabled = True):

                depth_features, depth_saliency_maps = model[1](depth)
                contours, saliency_maps = model[0](images, depth_features)

				loss = 0.0
				for i in contours:
					loss += structure_loss_contour(i, contour)
				
				for i in saliency_maps:
					loss += structure_loss_salient_fb(i, saliency, foreground=True)
					loss += salient_loss_weight * structure_loss_salient_fb(i, saliency, foreground=False)

                for i in depth_saliency_maps:
                    loss += structure_loss_salient_fb(i, saliency, foreground=True)
                    loss += salient_loss_weight * structure_loss_salient_fb(i, saliency, foreground=False)
			
			scaler.scale(loss).backward()

            scaler.step(optimizer[0])
            scaler.step(optimizer[1])
            scaler.update()

            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
			
			loss_end += loss.mean().item()
			count += 1
		if scheduler is not None:
            scheduler[0].step()
            scheduler[1].step()
		loss_before = loss_end/count
		print('[%d/%d]Loss: %.2f' % (epoch, epochs, loss_end/count), flush = True)
		if use_depth:
			mae = validation([model[0].eval(), model[1].eval()], test_loader, cuda, im_size, use_depth)
		else:
			mae = validation(model.eval(), test_loader, cuda, im_size, use_depth)
		if mae <= old_mae:
			old_mae = mae
		else:
			continue
		print("Saving Model!", flush = True)
        torch.save({
            'model_state_dict': model[0].module.state_dict()
        }, f = f_name + '_rgb' + '.pt')
        torch.save({
            'model_state_dict': model[1].module.state_dict()
        }, f = f_name + '_depth' + '.pt')
		
	return model

def main(
		train = False, 
		lr = 0.001, 
		epochs = 30, 
		saving_epoch = 25, 
		f_name = 'checkpoints/model.pt', 
		device_list = None, 
		device = 0, 
		batch = 0, 
		sched = 1,
		training_scheme = 'USOD', 
		loss_threshold = 0.0,
		salient_loss_weight = 1.0, 
		use_pretrained = 0, 
		checkpoint_name = 0, 
		im_size = None
	):


	cuda = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu")
	model_depth = USODFuseNet(1, 1, use_contour = False, factorw = 1, deep_supervision = True, dilation_rates = [[6, 10, 14, 18, 22], [6, 10, 14, 18]], encoder_only = True, use_depth = False); init_weights(model_depth)
	model = USODFuseNet(3, 1, use_contour = True, ssl = False, deep_supervision = True, factorw = 4, dilation_rates = [[6, 10, 14, 18, 22], [6, 10, 14, 18]]); init_weights(model)

	if training_scheme == 'USOD':
        train_set = USODAugmentedLoader(im_size=im_size); print(len(train_set), flush = True)
        test_dataset = USODTestLoader()

    elif training_scheme == 'RGBD':
        train_set = AugmentedRGBDSODLoader(im_size=im_size); print(len(train_set), flush = True)
        test_dataset = RGBDSODTestLoader()

    else:
        raise NotImplementedError

    train_dataloader = DataLoader(train_set, batch_size = batch, shuffle = True, num_workers = 8, drop_last = True)
	print("Length of Test Dataset %d" %(len(test_dataset)), flush = True)
	test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = True, num_workers = 8)

	if use_pretrained:

		assert checkpoint_name != '0'

		if os.path.exists('checkpoints/' + checkpoint_name + '.pt'):
			checkpoint_name = 'checkpoints/' + checkpoint_name + '.pt'
			print("Name of the Checkpoint - %s" %(checkpoint_name), flush = True)
			checkpoint = torch.load(checkpoint_name)
			model.load_state_dict(checkpoint['model_state_dict'], strict = False)
			if device_list is not None:
				model = nn.DataParallel(model, device_ids = device_list)
				if use_depth:
					model_depth = nn.DataParallel(model_depth, device_ids = device_list)
			else:
				print(device, flush = True)
			model.to(cuda)
			model_depth.to(cuda)
			print('Checkpoint Available!', flush = True)

		else:
			checkpoint_name_rgb = 'checkpoints/' + checkpoint_name + '_rgb.pt'
			checkpoint_name_depth = 'checkpoints/' + checkpoint_name + '_depth.pt'
			print("Name of the RGB Checkpoint - %s" %(checkpoint_name_rgb), flush = True)
			print("Name of the Depth Checkpoint - %s" %(checkpoint_name_depth), flush = True)
			checkpoint_rgb = torch.load(checkpoint_name_rgb)
			checkpoint_depth = torch.load(checkpoint_name_depth)
			model.load_state_dict(checkpoint_rgb['model_state_dict'], strict = True)
			model_depth.load_state_dict(checkpoint_depth['model_state_dict'], strict = True)
			if device_list is not None:
				model = nn.DataParallel(model, device_ids = device_list)
				if use_depth:
					model_depth = nn.DataParallel(model_depth, device_ids = device_list)
			else:
				print(device, flush = True)
			model.to(cuda)
			model_depth.to(cuda)
			print('Checkpoint Available!', flush = True)
	
	else:

		if device_list is not None:
			model = nn.DataParallel(model, device_ids = device_list)
			if use_depth:
				model_depth = nn.DataParallel(model_depth, device_ids = device_list)
		else:
			print(device, flush = True)
		
		model.to(cuda)
		if use_depth:
			model_depth.to(cuda)
		print(cuda, flush = True)
		print("Without Pre-Training", flush = True)

	optimizer = torch.optim.Adam(model.parameters(), lr = lr)
	optimizer_depth = torch.optim.Adam(model_depth.parameters(), lr = lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=saving_epoch, gamma=0.1)
    scheduler_depth = torch.optim.lr_scheduler.StepLR(optimizer_depth, step_size=saving_epoch, gamma=0.1)

	start = time.time()

	if train:
        model = training_deepx(
            epochs, saving_epoch, [model, model_depth], train_dataloader, test_loader, cuda, [optimizer, optimizer_depth], 
            [scheduler, scheduler_depth], f_name = f_name, loss_threshold = loss_threshold,
            salient_loss_weight = salient_loss_weight, im_size = im_size
        )
		end = time.time()
		print('Time taken for %d with a batch_size of %d is %.2f hours.' %(epochs, batch, (end - start) / (3600)), flush = True)

def parse_args():
	parser = argparse.ArgumentParser(description="Train a model with given hyperparameters.")

	parser.add_argument("--lr", 
						type=float, 
						default=0.001, 
						help="Learning rate.")
	parser.add_argument("--epochs", 
						type=int, 
						default=10, 
						help="Number of epochs to train.")
	parser.add_argument("--f_name", 
						type=str, 
						default="checkpoints/DefaultName", 
						help="Base folder name for checkpoints.")
	parser.add_argument("--n", 
						type=int, 
						default=1, 
						help="Number of devices (e.g., GPUs) to use.")
	parser.add_argument("--b", 
						type=int, 
						default=1, 
						help="Batch size per device.")
	parser.add_argument("--sched", 
						type=int, 
						default=1, 
						help="Use LR Scheduler.")
	parser.add_argument("--training_scheme", 
						type=str, 
						default="USOD", 
						help="Name of the training scheme. Can either be USOD or RGBD.")
	parser.add_argument("--salient_loss_weight", 
						type=float, 
						default=1.0, 
						help="Weight for salient loss.")
	parser.add_argument("--use_pretrained", 
						type=int, 
						default=1, 
						help="Whether to use pretrained weights (1) or not (0).")
	parser.add_argument("--checkpoint_name", 
						type=str, 
						default="DiffTesting", 
						help="Checkpoint filename (without folder).")
	parser.add_argument("--im_size", 
						type=int, 
						default=384, 
						help="Input image size.")

	return parser.parse_args()

def main(args):
	print(f"Learning Rate  = {args.lr}")
	print(f"Epochs         = {args.epochs}")
	print(f"Checkpoint dir = {args.f_name}")
	print(f"Devices (n)    = {args.n}")
	print(f"Batch per dev  = {args.b}")
	print(f"Use LR Scheduler  = {args.sched}")
	print(f"Train scheme   = {args.training_scheme}")
	print(f"Background Weight = {args.salient_loss_weight}")
	print(f"Use pretrained = {args.use_pretrained}")
	print(f"Checkpoint name  = {args.checkpoint_name}")
	print(f"Image size     = {args.im_size}")

	if args.n > 1:
		device_list = [i for i in range(args.n)]
		total_batch = args.n * args.b
		train_model(
			train=True,
			lr=args.lr,
			epochs=args.epochs,
			f_name=args.f_name,
			device_list=device_list,
			batch=total_batch,
			sched=args.sched,
			training_scheme=args.training_scheme,
			salient_loss_weight=args.salient_loss_weight,
			use_pretrained=args.use_pretrained,
			checkpoint_name=args.checkpoint_name,
			im_size=args.im_size,
			model_size=args.model_size
		)
	else:
		train_model(
			train=True,
			lr=args.lr,
			epochs=args.epochs,
			f_name=args.f_name,
			batch=args.b,
			sched=args.sched,
			training_scheme=args.training_scheme,
			salient_loss_weight=args.salient_loss_weight,
			use_pretrained=args.use_pretrained,
			checkpoint_name=args.checkpoint_name,
			im_size=args.im_size,
			model_size=args.model_size
		)

if __name__ == "__main__":
	seed_val = 60
	torch.manual_seed(seed_val)
	import random
	random.seed(seed_val)
	np.random.seed(seed_val)
	args = parse_args()
	main(args)
