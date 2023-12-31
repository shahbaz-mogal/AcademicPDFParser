"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Being reused by Shahbaz Mogal (shahbazmogal@outlook.com)
for educational purposes.
"""
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from PIL import Image
import torch
import torch.nn as nn
from torchvision.transforms.functional import resize, rotate
from PIL import ImageOps
import cv2
import numpy as np
import os
from typing import Optional, Union, List
import logging
from timm.models.swin_transformer import SwinTransformer
from academicpdfparser.transforms import train_transform, test_transform
import argparse
from pathlib import Path

class SwinEncoder(nn.Module):
    r"""
    Encoder based on SwinTransformer
    Set the initial weights and configuration with a pretrained SwinTransformer and then
    modify the detailed configurations

    Args:
    input_size: Input image size (width, height)
    align_long_axis: Whether to rotate image if height is greater than width
    """
    def __init__(
            self,
            input_size: List[int],
            align_long_axis: bool,
            ):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.model = SwinTransformer()
    
    @staticmethod
    def crop_margin(img: Image.Image) -> Image.Image:
        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        return img.crop((a, b, w + a, h + b))
    
    @staticmethod
    def crop_margin(img: Image.Image) -> Image.Image:
        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        return img.crop((a, b, w + a, h + b))

    @property
    def to_tensor(self):
        if self.training:
            return train_transform
        else:
            return test_transform

    def prepare_input(
        self, img: Image.Image, random_padding: bool = False
    ) -> torch.Tensor:
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        if img is None:
            return
        # crop margins
        try:
            img = self.crop_margin(img.convert("RGB"))
        except OSError:
            # might throw an error for broken files
            return
        if img.height == 0 or img.width == 0:
            return
        if self.align_long_axis and (
            (self.input_size[0] > self.input_size[1] and img.width > img.height)
            or (self.input_size[0] < self.input_size[1] and img.width < img.height)
        ):
            img = rotate(img, angle=-90, expand=True)
        img = resize(img, min(self.input_size))
        img.thumbnail((self.input_size[1], self.input_size[0]))
        delta_width = self.input_size[1] - img.width
        delta_height = self.input_size[0] - img.height
        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return self.to_tensor(ImageOps.expand(img, padding))
        

class AcademicPDFConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AcademicPDFModel`]. It is used to
    instantiate a Academic PDF Parser model according to the specified arguments, defining the model architecture

    Args:
        input_size:
            Input image size (canvas size) of Nougat.encoder, SwinTransformer in this codebase
        align_long_axis:
            Whether to rotate image if height is greater than width
    """
    model_type = "academicpdfparser"

    def __init__(
        self,
        input_size: List[int] = [896, 672],
        align_long_axis: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis

class AcademicPDFModel(PreTrainedModel):
    """
    Academic PDF Model
    The encoder converts an image of an academic document into a series of embeddings.
    Then, the decoder generates a sequence of tokens based on encoder's output.
    This sequence can be translated into a structured markup language format.
    """
    config_class = AcademicPDFConfig
    base_model_prefix = "academicpdfparser"
    def __init__(self, config: AcademicPDFConfig):
        super().__init__(config)
        self.config = config
        self.encoder = SwinEncoder(
            input_size=self.config.input_size,
            align_long_axis=self.config.align_long_axis,
        )

    def forward(
        self,
    ):
        pass

    def _init_weights(self, *args, **kwargs):
        return
    
    def inference(
        self,
        image: Image.Image = None,
        image_tensors: Optional[torch.Tensor] = None,
    ):
        """
        Generate a token sequence in an auto-regressive manner.

        Args:
            image: input document image (PIL.Image)
            image_tensors: (1, num_channels, height, width)
                convert prompt to tensor if image_tensor is not fed
        """
        output = {
            "predictions": list(),
        }
        if image is None and image_tensors is None:
            logging.warn("Image not found")
            return output

        if image_tensors is None:
            image_tensors = self.encoder.prepare_input(image).unsqueeze(0)
        
        if self.device.type != "mps":
            image_tensors = image_tensors.to(next(self.parameters()).dtype)

        image_tensors = image_tensors.to(self.device)
        print("Image Tensors type", type(image_tensors))
        # Print out the shape of the image tensors
        print("Image Tensors shape", image_tensors.shape)
        print("Image Tensors", image_tensors)
        return output
    

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, bytes, os.PathLike],
        *model_args,
        **kwargs,
    ):
        r"""
        Instantiate a pretrained nougat model from a pre-trained model configuration

        Args:
            model_path:
                Name of a pretrained model name either registered in huggingface.co. or saved in local.
        """
        model = super(AcademicPDFModel, cls).from_pretrained(
            model_path, *model_args, **kwargs
        )
        return model
    
# Delete this later: Used for testing purposes
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, help="Image", default=None)
    args = parser.parse_args()
    img_file = args.image
    print("Parsed arguments. Image File Name:", img_file)
    assert img_file.exists() and img_file.is_file()
    img = Image.open(img_file)
    print("Image Opened")
    config = AcademicPDFConfig()
    model = AcademicPDFModel(config)
    model.eval()
    output = model.inference(img)