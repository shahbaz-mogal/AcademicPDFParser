import logging
from typing import Callable
from academicpdfparser.dataset.rasterize import rasterize_paper
from functools import partial
from PIL import Image
import torch
import pypdf

class ImageDataset(torch.utils.data.Dataset):
    """
    Dataset for processing a list of images using a preparation function.

    This dataset takes a list of image paths and applies a preparation function to each image.

    Args:
        img_list (list): List of image paths.
        prepare (Callable): A preparation function to process the images.

    Attributes:
        img_list (list): List of image paths.
        prepare (Callable): The preparation function.
    """
    def __init__(self, img_list, prepare: Callable):
        super().__init__()
        self.img_list = img_list
        self.prepare = prepare

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.img_list[idx])
            return self.prepare(img)
        except Exception as e:
            logging.error(e)
    
    @staticmethod
    def ignore_none_collate(batch):
        if batch is None:
            return
        try:
            batch = [x for x in batch if x is not None and x[0] is not None]
            if len(batch) == 0:
                return
            return torch.utils.data.dataloader.default_collate(batch)
        except AttributeError:
            pass
    
class LazyDataset(torch.utils.data.Dataset):
    """
    Lazy loading dataset for processing PDF documents.

    This dataset allows lazy loading of PDF documents and provides access to processed images
    using a specified preparation function.

    Args:
        pdf (str): Path to the PDF document.
        prepare (Callable): A preparation function to process the images.

    Attributes:
        name (str): Name of the PDF document.
    """
    def __init__(self, pdf, prepare: Callable):
        super().__init__()
        self.prepare = prepare
        self.name = str(pdf)
        self.init_fn = partial(rasterize_paper, pdf)
        self.dataset = None
        self.size = len(pypdf.PdfReader(pdf).pages)

    def __len__(self):
        return self.size
    
    def __getitem__(self, i):
        """Return a page image, name from the PDF document"""
        if i == 0 or self.dataset is None:
            self.dataset = ImageDataset(self.init_fn(), self.prepare)
        if i <= self.size and i >= 0:
            return self.dataset[i], self.name if i == self.size - 1 else ""
        else:
            raise IndexError
        
    @staticmethod
    def ignore_none_collate(batch):
        if batch is None:
            return None, None
        try:
            _batch = []
            for i, x in enumerate(batch):
                image, name = x
                if image is not None:
                    _batch.append(x)
                elif name:
                    if i > 0:
                        _batch[-1] = (_batch[-1][0], name)
                    elif len(batch) > 1:
                        _batch.append((batch[1][0] * 0, name))
            if len(_batch) == 0:
                return None, None
            return torch.utils.data.dataloader.default_collate(_batch)
        except AttributeError:
            pass
        return None, None