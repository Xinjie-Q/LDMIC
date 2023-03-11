import argparse
import json
import math
import sys
import os
import time
import struct

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim
from torch import Tensor
from torch.cuda import amp
from torch.utils.model_zoo import tqdm
import compressai

from compressai.zoo.pretrained import load_pretrained
from compressai.zoo.image import model_urls, cfgs
from models.ldmic import *
from models.entropy_model import *
from model_zoo import models_arch
from torch.hub import load_state_dict_from_url
from lib.utils import CropCityscapesArtefacts, MinimalCrop


def collect_images(data_name:str, rootpath: str, num_camera:int):
    if data_name == 'cityscapes':
        left_image_list, right_image_list = [], []
        path = Path(rootpath)
        for left_image_path in path.glob(f'leftImg8bit/test/*/*.png'):
            left_image_list.append(str(left_image_path))
            right_image_list.append(str(left_image_path).replace("leftImg8bit", 'rightImg8bit'))

    elif data_name == 'instereo2k':
        path = Path(rootpath)
        path = path / "test"   
        folders = [f for f in path.iterdir() if f.is_dir()]
        left_image_list = [f / 'left.png' for f in folders]
        right_image_list = [f / 'right.png' for f in folders] #[1, 3, 860, 1080], [1, 3, 896, 1152]

    elif data_name == 'wildtrack':
        C1_image_list, C4_image_list = [], []
        path = Path(rootpath)
        for image_path in path.glob(f'images/C1/*.png'):
            if int(image_path.stem) > 2000:
                C1_image_list.append(str(image_path))
                C4_image_list.append(str(image_path).replace("C1", 'C4'))
        left_image_list, right_image_list = C1_image_list, C4_image_list

    elif data_name == 'multi_wildtrack':
        image_lists = [[] for i in range(num_camera)]
        path = Path(rootpath)
        for image_path in path.glob(f'images/C1/*.png'):
            if int(image_path.stem) > 2000:
                image_lists[0].append(str(image_path))
                for idx in range(1, num_camera):
                    image_lists[idx].append(str(image_path).replace("C1", 'C'+str(idx+1)))
        return image_lists
        

    return [left_image_list, right_image_list]


def aggregate_results(filepaths: List[Path]) -> Dict[str, Any]:
    metrics = defaultdict(list)
    # sum
    for f in filepaths:
        with f.open("r") as fd:
            data = json.load(fd)
        for k, v in data["results"].items():
            metrics[k].append(v)

    # normalize
    agg = {k: np.mean(v) for k, v in metrics.items()}
    return agg

def pad(x: Tensor, p: int = 2 ** (4 + 3)) -> Tuple[Tensor, Tuple[int, ...]]:
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    padding = (padding_left, padding_right, padding_top, padding_bottom)
    x = F.pad(x, padding, mode="constant", value=0)
    return x, padding


def crop(x: Tensor, padding: Tuple[int, ...]) -> Tensor:
    return F.pad(x, tuple(-p for p in padding))


def compute_metrics_for_frame(
    org_frame: Tensor,
    rec_frame: Tensor,
    device: str = "cpu",
    max_val: int = 255,):

    #psnr_float = -10 * torch.log10(F.mse_loss(org_frame, rec_frame))
    #ms_ssim_float = ms_ssim(org_frame, rec_frame, data_range=1.0)

    org_frame = (org_frame * max_val).clamp(0, max_val).round()
    rec_frame = (rec_frame * max_val).clamp(0, max_val).round()
    mse_rgb = (org_frame - rec_frame).pow(2).mean()
    psnr_float = 20 * np.log10(max_val) - 10 * torch.log10(mse_rgb)
    ms_ssim_float = ms_ssim(org_frame, rec_frame, data_range=max_val)

    return psnr_float, ms_ssim_float


def compute_bpp(likelihoods, num_pixels):
    bpp = sum(
        (torch.log(likelihood).sum() / (-math.log(2) * num_pixels))
        for likelihood in likelihoods.values()
    )
    return bpp


def read_image(crop_transform, filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    if crop_transform is not None:
        img = crop_transform(img)
    return transforms.ToTensor()(img)

@torch.no_grad()
def eval_model_entropy_estimation(IFrameCompressor:nn.Module, filepaths: List, **args: Any) -> Dict[str, Any]:
    device = next(IFrameCompressor.parameters()).device
    num_frames = len(filepaths[0])
    max_val = 2**8 - 1
    results = defaultdict(list)
    num_camera = args["num_camera"]
    print("cameras:", len(filepaths))
    if args["crop"]:
        crop_transform = CropCityscapesArtefacts() if args["data_name"] == "cityscapes" else MinimalCrop(min_div=64)
    else:
        crop_transform = None

    with tqdm(total=num_frames) as pbar: #97: 0-96
        for i in range(num_frames):

            x_list = []

            for f in filepaths:
                x = read_image(crop_transform, f[i]).unsqueeze(0).to(device)
                num_pixels = x.size(2) * x.size(3)
                x_list.append(x)

            out = IFrameCompressor(x_list)

            metrics = {}
            metrics["psnr-float"], metrics["ms-ssim-float"] = 0, 0
            metrics["bpp"] = 0
            for idx, f in enumerate(filepaths):
                x_rec = out["x_hat"][idx].clamp(0, 1)
                metrics[f"index{idx}-psnr-float"], metrics[f"index{idx}-ms-ssim-float"] = compute_metrics_for_frame(x_list[idx], x_rec, device, max_val)
                likelihoods = out["likelihoods"][idx]
                metrics[f"index{idx}-bpp"] = compute_bpp(likelihoods, num_pixels)
                metrics["bpp"] += metrics[f"index{idx}-bpp"]/num_camera

                metrics["psnr-float"] += metrics[f"index{idx}-psnr-float"]/num_camera
                metrics["ms-ssim-float"] += metrics[f"index{idx}-ms-ssim-float"]/num_camera

            for k, v in metrics.items():
                results[k].append(v)
            pbar.update(1)

    seq_results: Dict[str, Any] = {
        k: torch.mean(torch.stack(v)) for k, v in results.items()
    }
    for k, v in seq_results.items():
        if isinstance(v, torch.Tensor):
            seq_results[k] = v.item()
    return seq_results


def run_inference(
    filepaths,
    IFrameCompressor: nn.Module, 
    outputdir: Path,
    entropy_estimation: bool = False,
    trained_net: str = "",
    description: str = "",
    **args: Any):


    with amp.autocast(enabled=args["half"]):
        with torch.no_grad():
            if entropy_estimation:
                metrics = eval_model_entropy_estimation(IFrameCompressor, filepaths, **args)
    
    return metrics

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-view image compression network evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d", "--dataset", type=str, required=True, help="sequences directory")
    parser.add_argument("--data-name", type=str, required=True, help="sequences directory")
    parser.add_argument("--output", type=str, help="output directory")
    parser.add_argument(
        "-im",
        "--IFrameModel",
        default="Multi_LDMIC",
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument("-iq", "--IFrame_quality", type=int, default=4, help='Model quality')
    parser.add_argument("--i_model_path", type=str, help="Path to a checkpoint")
    parser.add_argument("--crop", action="store_true", help="use crop")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--half", action="store_true", help="use AMP")
    parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parser.add_argument(
        "--keep_binaries",
        action="store_true",
        help="keep bitstream files in output directory",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parser.add_argument("--metric", type=str, default="mse", help="metric: mse, ms-ssim")
    parser.add_argument("--cpu_num", type=int, default=4)
    parser.add_argument(
        "--num-camera", type=int, default=7, help="The number of cameras"
    )
    return parser


def main(args: Any = None) -> None:
    if args is None:
        args = sys.argv[1:]
    parser = create_parser()
    args = parser.parse_args(args)

    description = (
        "entropy-estimation" if args.entropy_estimation else args.entropy_coder
    )
    filepaths = collect_images(args.data_name, args.dataset, args.num_camera)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        raise SystemExit(1)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if device == "cpu":
        cpu_num = args.cpu_num # 这里设置成你想运行的CPU个数
        os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
        os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
        os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
        os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
        os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
        torch.set_num_threads(cpu_num)

    if args.IFrameModel == "Multi_LDMIC":
        IFrameCompressor = Multi_LDMIC(N=192, M=192, decode_atten=Multi_JointContextTransfer,)
    elif args.IFrameModel == "Multi_LDMIC_checkboard":
        IFrameCompressor = Multi_LDMIC_checkboard(N=192, M=192, decode_atten=Multi_JointContextTransfer,)
                      
    IFrameCompressor = IFrameCompressor.to(device)
    if args.i_model_path:
        print("Loading model:", args.i_model_path)
        checkpoint = torch.load(args.i_model_path, map_location=device)
        IFrameCompressor.load_state_dict(checkpoint["state_dict"])
        IFrameCompressor.update(force=True)
        IFrameCompressor.eval()

    # create output directory
    outputdir = args.output
    Path(outputdir).mkdir(parents=True, exist_ok=True)
    results = defaultdict(list)
    args_dict = vars(args)

    trained_net = f"{args.IFrameModel}-{args.metric}-{description}"
    metrics = run_inference(filepaths, IFrameCompressor, outputdir, trained_net=trained_net, description=description, **args_dict)
    for k, v in metrics.items():
        results[k].append(v)

    output = {
        "name": f"{args.IFrameModel}-{args.metric}",
        "description": f"Inference ({description})",
        "results": results,
    }

    with (Path(f"{outputdir}/{args.IFrameModel}-{args.metric}-{description}.json")).open("wb") as f:
        f.write(json.dumps(output, indent=2).encode())
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
