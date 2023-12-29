"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Being reused by Shahbaz Mogal (shahbazmogal@outlook.com)
for educational purposes.
"""
import argparse
from pathlib import Path
import logging
import sys
import pypdf
from academicpdfparser.utils.dataset import LazyDataset
import torch
from torch.utils.data import ConcatDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batchsize",
        "-b",
        type=int,
        default=default_batch_size(),
        help="Batch size to use.",
    )
    parser.add_argument("--out", "-o", type=Path, help="Output directory.")
    parser.add_argument("pdf", nargs="+", type=Path, help="PDF(s) to process.")
    args = parser.parse_args()

    if args.out is None:
        logging.warning("No output directory. Output will be printed to console.")
    else:
        if not args.out.exists():
            logging.info("Output directory does not exist. Creating output directory.")
            args.out.mkdir(parents=True)
        if not args.out.is_dir():
            logging.error("Output has to be directory.")
            sys.exit(1)

    if len(args.pdf) == 1 and not args.pdf[0].suffix == ".pdf":
        # input is a list of pdfs. Else arg.pdf is already in correct format
        try:
            pdfs_path = args.pdf[0]
            if pdfs_path.is_dir():
                args.pdf = list(pdfs_path.rglob("*.pdf"))
            else:
                args.pdf = [
                    Path(l) for l in open(pdfs_path).read().split("\n") if len(l) > 0
                ]
            logging.info(f"Found {len(args.pdf)} files.")
        except:
            pass
    return args

def main():
    args = get_args()
    datasets = []
    for pdf in args.pdf:
        if not pdf.exists():
            continue
        if args.out:
            out_path = args.out / pdf.with_suffix(".mmd").name
            if out_path.exists() and not args.recompute:
                logging.info(
                    f"Skipping {pdf.name}, already computed. Run with --recompute to convert again."
                )
                continue
        try:
            dataset = LazyDataset(pdf)
        except pypdf.errors.PdfStreamError:
            logging.info(f"Could not load file {str(pdf)}.")
            continue
        datasets.append(dataset)
    if len(datasets) == 0:
        return
    dataloader = torch.utils.data.DataLoader(
        ConcatDataset(datasets),
        batch_size=args.batchsize,
        shuffle=False,
        collate_fn=LazyDataset.ignore_none_collate,
    )
    

if __name__ == "__main__":
    main()