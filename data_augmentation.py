import os
import sys
import random
import argparse
import hashlib
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from tqdm import tqdm


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def is_image_file(filename):
	return os.path.splitext(filename.lower())[1] in IMAGE_EXTENSIONS


def random_crop_and_resize(img, min_scale=0.85):
	w, h = img.size
	scale = random.uniform(min_scale, 1.0)
	new_w, new_h = int(w * scale), int(h * scale)
	if new_w == w and new_h == h:
		return img
	left = random.randint(0, w - new_w)
	top = random.randint(0, h - new_h)
	box = (left, top, left + new_w, top + new_h)
	cropped = img.crop(box)
	return cropped.resize((w, h), Image.LANCZOS)


def add_gaussian_noise(np_img, sigma=8.0):
	noise = np.random.normal(0, sigma, np_img.shape).astype(np.float32)
	out = np_img.astype(np.float32) + noise
	out = np.clip(out, 0, 255)
	return out


def apply_random_augmentations(img, seed=None):
	"""Apply a random chain of augmentations to a PIL Image and return (aug_image, ops_str)."""
	if seed is not None:
		random.seed(seed)
		np.random.seed(seed & 0xFFFFFFFF)

	ops = []

	# Horizontal flip
	if random.random() < 0.5:
		img = img.transpose(Image.FLIP_LEFT_RIGHT)
		ops.append('hflip')

	# Vertical flip (less common)
	if random.random() < 0.15:
		img = img.transpose(Image.FLIP_TOP_BOTTOM)
		ops.append('vflip')

	# Rotation
	if random.random() < 0.8:
		angle = random.uniform(-25, 25)
		img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
		ops.append(f'rot{int(angle)}')

	# Random crop + resize
	if random.random() < 0.6:
		img = random_crop_and_resize(img, min_scale=0.85)
		ops.append('crop')

	# Brightness
	if random.random() < 0.95:
		factor = random.uniform(0.75, 1.30)
		img = ImageEnhance.Brightness(img).enhance(factor)
		ops.append(f'bright{round(factor,2)}')

	# Contrast
	if random.random() < 0.9:
		factor = random.uniform(0.80, 1.25)
		img = ImageEnhance.Contrast(img).enhance(factor)
		ops.append(f'cont{round(factor,2)}')

	# Blur
	if random.random() < 0.3:
		radius = random.uniform(0.0, 1.8)
		img = img.filter(ImageFilter.GaussianBlur(radius))
		ops.append(f'blur{round(radius,2)}')

	# Noise
	if random.random() < 0.4:
		np_img = np.array(img).astype(np.float32)
		np_img = add_gaussian_noise(np_img, sigma=random.uniform(4.0, 12.0))
		img = Image.fromarray(np_img.astype(np.uint8))
		ops.append('noise')

	ops_str = '_'.join(ops) if ops else 'orig'
	return img, ops_str


def process_dataset(src_root, dst_root, augment_per_image=5, copy_original=False, seed=None):
	"""Walk `src_root`, apply augmentations and save into `dst_root` mirroring subfolders."""
	src_root = os.path.abspath(src_root)
	dst_root = os.path.abspath(dst_root)
	if not os.path.isdir(src_root):
		raise ValueError(f"Source directory not found: {src_root}")

	os.makedirs(dst_root, exist_ok=True)

	# Collect all image paths so we can show a progress bar
	all_image_paths = []
	for root, dirs, files in os.walk(src_root):
		for f in files:
			if is_image_file(f):
				all_image_paths.append(os.path.join(root, f))

	for in_path in tqdm(all_image_paths, desc='Images', unit='img'):
		try:
			rel_dir = os.path.relpath(os.path.dirname(in_path), src_root)
			dst_dir = os.path.join(dst_root, rel_dir) if rel_dir != '.' else dst_root
			os.makedirs(dst_dir, exist_ok=True)

			with Image.open(in_path) as im:
				img = im.convert('RGB')

			base, ext = os.path.splitext(os.path.basename(in_path))

			if copy_original:
				out_name = f"{base}_orig{ext}"
				out_path = os.path.join(dst_dir, out_name)
				try:
					img.save(out_path, quality=95)
				except Exception:
					img.save(out_path)

			for i in range(augment_per_image):
				aug_seed = None if seed is None else (seed + i)
				img_aug, ops = apply_random_augmentations(img, seed=aug_seed)
				uniq = hashlib.sha1(os.urandom(8)).hexdigest()[:6]
				safe_ops = ops.replace(' ', '').replace('/', '-')
				out_name = f"{base}_aug{i+1}_{safe_ops}_{uniq}{ext}"
				out_path = os.path.join(dst_dir, out_name)
				try:
					img_aug.save(out_path, quality=95)
				except Exception:
					img_aug.save(out_path)

		except Exception as e:
			print(f"Failed to process {in_path}: {e}", file=sys.stderr)


def build_arg_parser():
	p = argparse.ArgumentParser(description='Simple image augmentation script for disease dataset')
	p.add_argument('--src', '-s', default='disease dataset', help='Source dataset root directory')
	p.add_argument('--dst', '-d', default='augmented_dataset', help='Destination root for augmented images')
	p.add_argument('--n', '-n', type=int, default=5, help='Augmentations to create per image')
	p.add_argument('--copy-original', action='store_true', help='Copy original images into destination as well')
	p.add_argument('--seed', type=int, default=None, help='Optional random seed for reproducibility')
	return p


def main(argv=None):
	argv = argv if argv is not None else sys.argv[1:]
	parser = build_arg_parser()
	args = parser.parse_args(argv)

	print(f"Source: {args.src}")
	print(f"Destination: {args.dst}")
	print(f"Augmentations per image: {args.n}")
	if args.seed is not None:
		print(f"Seed: {args.seed}")

	process_dataset(args.src, args.dst, augment_per_image=args.n, copy_original=args.copy_original, seed=args.seed)


if __name__ == '__main__':
	main()

