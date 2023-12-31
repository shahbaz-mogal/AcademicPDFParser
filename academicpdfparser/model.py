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
import os
from typing import Optional, Union

class AcademicPDFConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AcademicPDFModel`]. It is used to
    instantiate a Academic PDF Parser model according to the specified arguments, defining the model architecture

    Args:
        
    """
    model_type = "academicpdfparser"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

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