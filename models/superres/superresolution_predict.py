import argparse
# from importlib.resources import path
from unittest import result
import cv2
import glob
import os
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from PIL import Image
import numpy as np

from utils.model_download import ensure_file

# The 64 MB weights are not tracked in git (docs/AUDIT.md H-6): fetched on first use, see weights_info.py
from models.superres.weights_info import WEIGHTS_PATH, WEIGHTS_SHA256, WEIGHTS_URL


"""
The follwoing class can either be used from other scripts or can be used directly in the terminal by giving the input path to the images, 
with the desired output path that the script will generate and fill with the super-res images.
The output will have 4 times the res of the initial image.
"""


class RealesrganSuperresolution:
    #Initianlize model before prediction/image-enhancement
    def __init__(self):
        model_path = ensure_file(WEIGHTS_PATH, WEIGHTS_URL, WEIGHTS_SHA256, 'Real-ESRGAN x4plus weights')



        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=256,
            tile_pad=10,
            pre_pad=0,
            # Disable CUDA/fp16 due to Blackwell GPU incompatibility with current PyTorch
            half=False)

    #Input: image in numnpy array format, returns the #upscaled-superres image numpy array format
    def enhance(self,img_np):
        output, _ = self.upsampler.enhance(img_np, outscale=4)
        return output

    #Input: input-path of the original images and output-path of the super-res images.
    def enhance_dir(self, input_dir, output_dir):
        for img in os.listdir(input_dir):
            # img_x=Image.open(os.path.join(input_dir,img))
            img_x=cv2.imread(os.path.join(input_dir,img),1)
            # img_x=np.array(img_x)
            img_superres=self.enhance(img_x)
            cv2.imwrite(os.path.join(output_dir,img), img_superres)
