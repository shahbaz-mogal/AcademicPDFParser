from torch.utils.data import Dataset
import pypdf
from academicpdfparser.dataset.rasterize import rasterize_paper

class PDFImageDataset(Dataset):
    """
    Lazy loading dataset for processing PDF documents.

    Args:
        pdf (str/Path): Path to the PDF document.

    Attributes:
        name (str): Name of the PDF document.
    """
    def __init__(self, pdf: str):
        super().__init__()
        self.name = self(pdf)
        self.img_list = rasterize_paper(pdf)
        self.size = len(self.img_list)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # TODO: Implement this
        raise IndexError("Not implemented yet")