import os
import argparse
from PIL import Image
from USODFuseNet import *
from torchvision import transforms
import tqdm
import torch
import torch.nn as nn
import numpy as np
import cv2 as cv

MODEL_WIDTH, MODEL_HEIGHT = 256, 256
# MODEL_WIDTH, MODEL_HEIGHT = 416, 416

def load_model(cuda=None):
	print("Loading model...")
	
	model_depth = USODFuseNet(1, 1, use_contour = False, factorw = 1, deep_supervision = True, dilation_rates = [[6, 10, 14, 18, 22], [6, 10, 14, 18]], encoder_only = True, use_depth = False)
	model_rgb = USODFuseNet(3, 1, use_contour = True, ssl = False, deep_supervision = True, factorw = 4, dilation_rates = [[6, 10, 14, 18, 22], [6, 10, 14, 18]])
	
	model_rgb.to(cuda)
	model_depth.to(cuda)

	checkpoint_rgb = torch.load('checkpoints_new/USODFuseNetRGBD_rgb.pt', map_location=cuda)
	checkpoint_depth = torch.load('checkpoints_new/USODFuseNetRGBD_depth.pt', map_location=cuda)

	model_rgb.load_state_dict(checkpoint_rgb['model_state_dict'], strict = True)
	model_depth.load_state_dict(checkpoint_depth['model_state_dict'], strict = True)

	model_rgb.eval()
	model_depth.eval()
	return model_depth, model_rgb

def run_inference(model_d, model_r, image, depth_image, cuda):
	original_width, original_height = image.size

	input_resized = transforms.Compose([transforms.Resize((MODEL_HEIGHT, MODEL_WIDTH)), transforms.ToTensor()])(image)
	depth_resized = transforms.Compose([transforms.Resize((MODEL_HEIGHT, MODEL_WIDTH)), transforms.ToTensor()])(depth_image)

	depth_features, _ = model_d(depth_resized.unsqueeze(0).to(cuda))
	_, prediction = model_r(input_resized.unsqueeze(0).to(cuda), depth_features)
	prediction = prediction[-1]
	pred = torch.sigmoid(prediction)
	pred = nn.Upsample(size=(original_height, original_width), mode='bilinear',align_corners=False)(pred)
	output_final = transforms.ToPILImage()(pred.squeeze(0))

	return output_final

def run_inference_folder(model_d, model_r, image, depth_image, cuda):
	_, h, w = image.size()

	input_resized = transforms.Compose([transforms.ToPILImage(), transforms.Resize((MODEL_HEIGHT, MODEL_WIDTH)), transforms.ToTensor()])(image)
	depth_resized = transforms.Compose([transforms.ToPILImage(), transforms.Resize((MODEL_HEIGHT, MODEL_WIDTH)), transforms.ToTensor()])(depth_image)

	depth_features, _ = model_d(depth_resized.unsqueeze(0).to(cuda))
	_, prediction = model_r(input_resized.unsqueeze(0).to(cuda), depth_features)
	prediction = prediction[-1]
	pred = torch.sigmoid(prediction)
	pred = nn.Upsample(size=(h, w), mode='bilinear',align_corners=False)(pred)
	pred /= torch.max(pred)
	smap = pred.float().squeeze().detach().cpu().numpy()
	smap *= 255
	smap = smap.astype(np.uint8)

	return smap

def process_single_image(model_d, model_r, input_path, depth_path, cuda, display=False):

	if not os.path.isfile(input_path):
		raise ValueError(f"Single image mode: '{input_path}' is not a valid file.")

	image = Image.open(input_path)
	depth_image = Image.open(depth_path).convert('L')
	prediction = run_inference(model_d, model_r, image, depth_image, cuda)

	if display:
		prediction.show()
	else:
		# Save with the name of the image followed by '_prediction'
		base_name = os.path.splitext(os.path.basename(input_path))[0]
		save_name = f"{base_name}_prediction.png"
		prediction.save(save_name)
		print(f"Saved prediction to '{save_name}'")

def process_folder(model_d, model_r, input_path, depth_path, output_dir, cuda):
	"""
	Process all images in a directory. Saves predictions in 'output_dir'.
	"""
	if not os.path.exists(input_path) or not os.path.isdir(input_path):
		raise ValueError(f"Folder mode: '{input_path}' is not a valid directory.")

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	input_dir = list(sorted(os.listdir(input_path)))
	depth_dir = list(sorted(os.listdir(depth_path)))

	for file_name, depth_file_name in tqdm.tqdm(zip(input_dir, depth_dir), total=len(input_dir)):
		file_path = os.path.join(input_path, file_name)
		depth_file_path = os.path.join(depth_path, depth_file_name)
		
		if (os.path.isfile(file_path) and file_name.endswith((".jpg", ".jpeg", ".png"))) \
		and \
		(os.path.isfile(depth_file_path) and depth_file_name.endswith((".jpg", ".jpeg", ".png"))) \
		:
			# print(file_path, depth_path, file_name, depth_file_name)
			inp_img = cv.imread(file_path)
			inp_img = cv.cvtColor(inp_img, cv.COLOR_BGR2RGB)
			inp_img = inp_img.astype('float32')

			inp_img /= np.max(inp_img)
			inp_img = np.transpose(inp_img, axes=(2, 0, 1))
			inp_img = torch.from_numpy(inp_img).float()

			depth_img = cv.imread(depth_file_path, 0)
			depth_img = depth_img.astype('float32')
			depth_img /= np.max(depth_img)
			depth_img = np.expand_dims(depth_img, axis=0)
			depth_img = torch.from_numpy(depth_img).float()

			# image = Image.open(file_path)
			# depth_image = Image.open(depth_file_path).convert('L')
			prediction = run_inference_folder(model_d, model_r, inp_img, depth_img, cuda)

			base_name = os.path.splitext(file_name)[0]
			out_file = f"{base_name}.png"
			out_path = os.path.join(output_dir, out_file)

			# prediction.save(out_path)
			cv.imwrite(out_path, prediction)

def main():
	parser = argparse.ArgumentParser(description="Run model inference on an image or a folder of images.")
	
	parser.add_argument(
		"--mode",
		type=str,
		choices=["single", "folder"],
		required=True,
		help="Mode of operation: 'single' for a single image, 'folder' for a directory of images."
	)
	parser.add_argument(
		"--input_path",
		type=str,
		required=True,
		help="Path to a single image (if mode=single) or to a folder (if mode=folder)."
	)
	parser.add_argument(
		"--depth_path",
		type=str,
		required=True,
		help="Path to a single image (if mode=single) or to a folder (if mode=folder)."
	)
	parser.add_argument(
		"--display",
		action="store_true",
		help="(Only for single image mode) Display the prediction instead of saving it."
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default=None,
		help="(Only for folder mode) Directory to save the output predictions."
	)

	args = parser.parse_args()
	device = 0 ## cpu
	cuda = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu")

	print(f"Mode         = {args.mode}")
	print(f"Input Path = {args.input_path}")
	print(f"Depth Path = {args.depth_path}")
	print(f"Output Directory   = {args.output_dir}")

	model_depth, model_rgb = load_model(cuda)

	with torch.no_grad():
		if args.mode == "single":
			process_single_image(model_depth, model_rgb, args.input_path, args.depth_path, cuda, display=args.display)
		elif args.mode == "folder":
			if not args.output_dir:
				raise ValueError("In folder mode, --output_dir is required.")
			process_folder(model_depth, model_rgb, args.input_path, args.depth_path, args.output_dir, cuda)

if __name__ == "__main__":
	main()
