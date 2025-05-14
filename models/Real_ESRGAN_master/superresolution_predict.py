import argparse
import os
import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact




class RealesrganSuperresolution:
    """
    The following class can either be used from other scripts or can be used directly in the derminal by giving the input path to the images, with the desired output path that the script will generate and fill with the super-res images.
    Initialize model before prediction/image-enhancement
    """
    print(os.listdir())
    def __init__(self):
        model_path = "RealESRGAN_x4plus.pth"

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=256,
            tile_pad=10,
            pre_pad=0,
            half="store_true",
        )


    def enhance(self, img_np):
        """Input: image in numnpy array format, returns the upscaled-superres image in a numpy array format"""
        output, _ = self.upsampler.enhance(img_np, outscale=4)
        return output

    def enhance_dir(self, input_dir, output_dir):
        """Input: input-path of the original images and output-path of the super-res images."""
        for img in os.listdir(input_dir):
            img_x = cv2.imread(os.path.join(input_dir, img), 1)
            img_superres = self.enhance(img_x)
            cv2.imwrite(os.path.join(output_dir, img), img_superres)

model_superres=RealesrganSuperresolution()
model_superres.enhance_dir('../../data/images2','../../data/images_superres')

# # if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="RealESR-GAN+ Super Resolution image upscaler"
#     )
#     parser.add_argument(
#         "-i", "--input", help="Path to input image directory", required=True
#     )
#     parser.add_argument(
#         "-o",
#         "--output",
#         help="Path to output image directory  - the script will generate the directory",
#         required=True,
#     )
#     args = vars(parser.parse_args())

#     if args["input"] and args["output"]:

#         check_input_path = os.path.isdir(args["input"])
#         check_output_path = os.path.isdir(args["output"])

#         if check_input_path and check_output_path:

#             try:
#                 os.mkdir(args["output"])
#                 superres_model = RealesrganSuperresolution()
#                 superres_model.enhance_dir(args["input"], args["output"])

#             except Exception:

#                 if os.path.isdir(args["output"]) is not None:
#                     print("Output directory with the same already exists")

#                 else:
#                     print("Wrong output path format/No images in input directory")

#     else:
#         print("Input- and output-path required")
