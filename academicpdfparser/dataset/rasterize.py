"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from pathlib import Path
import io
from typing import Optional, List
import pypdfium2
import logging
from tqdm import tqdm
import argparse

logging.getLogger("pypdfium2").setLevel(logging.WARNING)

def rasterize_paper(
        pdf: Path,
        outpath: Optional[Path] = None,
        dpi: int = 96,
        return_pil=False,
) -> Optional[List[io.BytesIO]]:
    """
    Rasterize a PDF file to PNG images.

    Args:
        pdf (str/Path): The path to the PDF file.
        outpath (Optional[Path], optional): The output directory. If None, the PIL images will be returned instead. Defaults to None.
        dpi (int, optional): The output DPI. Defaults to 96.
        return_pil (bool, optional): Whether to return the PIL images instead of writing them to disk. Defaults to False.

    Returns:
        Optional[List[io.BytesIO]]: The PIL images if `return_pil` is True, otherwise None.
    """
    pils = []
    if outpath is None:
        return_pil = True
    try:
        if isinstance(pdf, (str, Path)):
            pdf = pypdfium2.PdfDocument(pdf)
        PIL_images = pdf.render(
            pypdfium2.PdfBitmap.to_pil,
            scale=dpi / 72,
        )
        counter = 1
        for image in PIL_images:
            if return_pil:
                page_bytes = io.BytesIO()
                image.save(page_bytes, "bmp")
                pils.append(page_bytes)
            else:
                image.save((outpath / ("%02d.png" % counter)), "png")
                counter += 1
    except Exception as e:
        logging.error(f"Error rasterizing PDF {pdf}: {e}")
    if return_pil:
        return pils
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdfs", nargs="+", type=Path, help="PDF files", required=True)
    parser.add_argument("--out", type=Path, help="Output dir", default=None)
    parser.add_argument(
        "--dpi", type=int, default=96, help="What resolution the pages will be saved"
    )
    args = parser.parse_args()
    for pdf_file in tqdm(args.pdfs):
        assert pdf_file.exists() and pdf_file.is_file()
        outpath: Path = args.out or (pdf_file.parent / pdf_file.stem)
        outpath.mkdir(exist_ok=True)
        rasterize_paper(pdf_file, outpath, dpi=args.dpi)
