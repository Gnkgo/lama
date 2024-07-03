#!/usr/bin/env python3

import glob
import os
import shutil
import traceback

import PIL.Image as Image
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from saicinpainting.evaluation.masks.mask import SegmentationMask, propose_random_square_crop
print("Imported SegmentationMask and propose_random_square_crop successfully")
from saicinpainting.evaluation.utils import load_yaml, SmallMode
from saicinpainting.training.data.masks import MixedMaskGenerator
from saicinpainting.training.data.masks import FixedMaskGenerator


class MakeManyMasksWrapper:
    def __init__(self, impl, variants_n=2):
        self.impl = impl
        self.variants_n = variants_n

    def get_masks(self, img):
        img = np.transpose(np.array(img), (2, 0, 1))
        return [self.impl(img)[0] for _ in range(self.variants_n)]

class MakeManyMasksWrapperFixed:
    def __init__(self, impl):
        self.impl = impl

    def get_masks(self, img, index):
       #plt.imshow(img)   
       #plt.show() 

        img = np.transpose(np.array(img), (2, 0, 1))   
        # print("ATTENTION; index", index)
        # print("ATTENTION; img", img.shape)
        # print("ATTENTION; impl", self.impl)  
        # #show the image
        
        # print("ATTENTION; impl", self.impl(img, index)[0].shape)
        # for i in range(self.impl(img, index)[0].shape[0]):
        #     plt.imshow(self.impl(img, index)[0][i])
        #     plt.show()
        return [self.impl(img, index)[0]]
    
def process_images(src_images, indir, outdir, config):
    # if config.generator_kind == 'segmentation':
    #     mask_generator = SegmentationMask(**config.mask_generator_kwargs)
    # if config.generator_kind == 'random':
    #     variants_n = config.mask_generator_kwargs.pop('variants_n', 2)
    #     mask_generator = MakeManyMasksWrapper(MixedMaskGenerator(**config.mask_generator_kwargs),
    #                                           variants_n=variants_n)
    #if config.generator_kind == 'fixed' or config.generator_kind == 'segmentation':
    mask_generator = MakeManyMasksWrapperFixed(FixedMaskGenerator())
    # else:
    #     raise ValueError(f'Unexpected generator kind: {config.generator_kind}')

    max_tamper_area = config.get('max_tamper_area', 1)

    for infile in src_images:
        try:
            file_relpath = infile[len(indir):]
            img_outpath = os.path.join(outdir, file_relpath)
            os.makedirs(os.path.dirname(img_outpath), exist_ok=True)

            image = Image.open(infile).convert('RGB')

            # scale input image to output resolution and filter smaller images
            if min(image.size) < config.cropping.out_min_size:
                handle_small_mode = SmallMode(config.cropping.handle_small_mode)
                if handle_small_mode == SmallMode.DROP:
                    continue
                elif handle_small_mode == SmallMode.UPSCALE:
                    factor = config.cropping.out_min_size / min(image.size)
                    out_size = (np.array(image.size) * factor).round().astype('uint32')
                    image = image.resize(out_size, resample=Image.BICUBIC)
            else:
                factor = config.cropping.out_min_size / min(image.size)
                out_size = (np.array(image.size) * factor).round().astype('uint32')
                image = image.resize(out_size, resample=Image.BICUBIC)

            # generate and select masks
            if config.generator_kind == 'fixed': 
                # print("FIIIIIIIIIIIXEEEEEEEEEEEED")
                index = os.path.basename(infile).split('.')[0]
                # print("index: process_image ", index)
                # Extract the image shape and pass to the mask generator
                #image_shape = (image.height, image.width)
                
                src_masks = mask_generator.get_masks(image, index)
                # print("index: ", index) 
                # print("src_masks: ", src_masks)
                #print("-----------------src_masks has 1: ",  np.any(src_masks > 1))

                

            else:
                # print("--------------noooooooooooooot fixeeeeeeed")
                src_masks = mask_generator.get_masks(image)
            #show if src_mask has 1
            
            # print("source mask: ", src_masks)
            filtered_image_mask_pairs = []
            for cur_mask in src_masks:
                if config.cropping.out_square_crop:
                    (crop_left,
                     crop_top,
                     crop_right,
                     crop_bottom) = propose_random_square_crop(cur_mask,
                                                               min_overlap=config.cropping.crop_min_overlap)
                    cur_mask = cur_mask[crop_top:crop_bottom, crop_left:crop_right]
                    cur_image = image.copy().crop((crop_left, crop_top, crop_right, crop_bottom))
                else:
                    # print("cur_mask in else statement: ", cur_mask.shape)
                    cur_image = image


                # print("cur_mask: ", cur_mask.shape)
                # print("cur_image: ", cur_image.size)
                
                #show the mask and the image
                # plt.imshow(cur_mask)
                # plt.xlabel("mask")
                # plt.show()
                # plt.imshow(cur_image)
                # plt.xlabel("image")
                # plt.show()
                # print("cur_mask: ", cur_mask)
                
                if len(np.unique(cur_mask)) == 0:# or cur_mask.mean() > max_tamper_area:
                    continue
                
                # print("cur_mask: ", cur_mask.shape)
                # print("unique: ", np.unique(cur_mask))
                


                filtered_image_mask_pairs.append((cur_image, cur_mask))

            
            
            mask_indices = np.random.choice(len(filtered_image_mask_pairs),
                                            size=min(len(filtered_image_mask_pairs), config.max_masks_per_image),
                                            replace=False)

            print(f'Processing {infile} with {len(filtered_image_mask_pairs)} masks, ')
            # crop masks; save masks together with input image
            mask_basename = os.path.join(outdir, os.path.splitext(file_relpath)[0])
            # print("--------------------- before saving")
            for i, idx in enumerate(mask_indices):
                cur_image, cur_mask = filtered_image_mask_pairs[idx]
                cur_basename = mask_basename + f'_crop{i:03d}'
                #cur_mask = np.clip(cur_mask, 0, 1)
                # plt.imshow(cur_mask)
                # plt.xlabel("mask")
                # plt.show()
                Image.fromarray(np.clip(cur_mask * 255, 0, 255).astype('uint8'),
                                mode='L').save(cur_basename + f'_mask{i:03d}.png')
                cur_image.save(cur_basename + '.png')
                print("--------------------- after saving")
        except KeyboardInterrupt:
            return
        except Exception as ex:
            print(f'Could not make masks for {infile} due to {ex}:\n{traceback.format_exc()}')


def main(args):
    if not args.indir.endswith('/'):
        args.indir += '/'

    os.makedirs(args.outdir, exist_ok=True)

    config = load_yaml(args.config)

    in_files = list(glob.glob(os.path.join(args.indir, '**', f'*.{args.ext}'), recursive=True))
    if args.n_jobs == 0:
        process_images(in_files, args.indir, args.outdir, config)
    else:
        in_files_n = len(in_files)
        chunk_size = in_files_n // args.n_jobs + (1 if in_files_n % args.n_jobs > 0 else 0)
        Parallel(n_jobs=args.n_jobs)(
            delayed(process_images)(in_files[start:start+chunk_size], args.indir, args.outdir, config)
            for start in range(0, len(in_files), chunk_size)
        )


if __name__ == '__main__':
    import argparse    
    aparser = argparse.ArgumentParser()
    aparser.add_argument('config', type=str, help='Path to config for dataset generation')
    aparser.add_argument('indir', type=str, help='Path to folder with images')
    aparser.add_argument('outdir', type=str, help='Path to folder to store aligned images and masks to')
    aparser.add_argument('--n-jobs', type=int, default=0, help='How many processes to use')
    aparser.add_argument('--ext', type=str, default='png', help='Input image extension')
    #aparser.add_argument('--rectangles', type=str, help='Path to rectangles file (for fixed mask generator)')

    main(aparser.parse_args())

#python bin/gen_mask_dataset.py configs/data_gen/fixed.yaml my_dataset/val_source/ my_dataset/val/fixed.yaml --ext png --rectangles test.json