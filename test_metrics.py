# # Image Quality Assessment Script
# # Evaluates metrics. Compatible with Reference (PSNR, SSIM, etc.) and No-Reference modes.
# import pyiqa
# # print(pyiqa.list_models()) # Optional: Print available models
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# import sys
# import glob
# import argparse
# import logging
# from datetime import datetime
# import time

# import cv2
# import numpy as np
# import torch

# from basicsr.utils import img2tensor

# def get_timestamp():
#     """Returns the current timestamp in a specific format."""
#     return datetime.now().strftime('%y%m%d-%H%M%S')

# def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
#     """Sets up a logger with specified configurations."""
#     logger = logging.getLogger(logger_name)
#     formatter = logging.Formatter(
#         fmt='%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
#         datefmt='%y-%m-%d %H:%M:%S'
#     )
#     logger.setLevel(level)

#     if tofile:
#         log_file = os.path.join(root, f"{phase}_{get_timestamp()}.log")
#         fh = logging.FileHandler(log_file, mode='w')
#         fh.setFormatter(formatter)
#         logger.addHandler(fh)

#     if screen:
#         sh = logging.StreamHandler()
#         sh.setFormatter(formatter)
#         logger.addHandler(sh)

# def dict2str(opt, indent=1):
#     """Converts a dictionary to a formatted string for logging."""
#     msg = ''
#     for k, v in opt.items():
#         if isinstance(v, dict):
#             msg += ' ' * (indent * 2) + f"{k}:[\n"
#             msg += dict2str(v, indent + 1)
#             msg += ' ' * (indent * 2) + "]\n"
#         else:
#             msg += ' ' * (indent * 2) + f"{k}: {v}\n"
#     return msg

# def main():
#     parser = argparse.ArgumentParser(description="Image Quality Assessment Script")

#     parser.add_argument(
#         "--inp_imgs",
#         nargs="+",
#         required=True,
#         help="Path(s) to the input (SR) images directories."
#     )

#     # Modified: gt_imgs is now optional
#     parser.add_argument(
#         "--gt_imgs",
#         nargs="+",
#         required=False, 
#         default=None,
#         help="Path(s) to the ground truth (GT) images directories. Optional."
#     )

#     parser.add_argument(
#         "--log",
#         type=str,
#         required=True,
#         help="Directory path to save the log files."
#     )

#     parser.add_argument(
#         "--log_name",
#         type=str,
#         default='METRICS',
#         help="Base name for the log files."
#     )

#     args = parser.parse_args()

#     # Determine if GT is provided
#     has_gt = args.gt_imgs is not None

#     # Validate inputs if GT is provided
#     if has_gt:
#         if len(args.inp_imgs) != len(args.gt_imgs):
#             print("Error: The number of input image directories and GT image directories must be the same.")
#             sys.exit(1)
#     else:
#         # Create a dummy list of Nones for gt_imgs to simplify the loop later
#         args.gt_imgs = [None] * len(args.inp_imgs)

#     # Set device
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#     # Create log directory if it doesn't exist
#     os.makedirs(args.log, exist_ok=True)

#     # Initialize logger
#     try:
#         args.log_name = args.inp_imgs[0].split('/')[-1] # Changed split index to -1 for safety
#     except IndexError:
#         args.log_name = 'METRICS'
    
#     phase_name = f'test_{args.log_name}' if has_gt else f'test_NR_{args.log_name}'
#     setup_logger('base', args.log, phase_name, level=logging.INFO, screen=True, tofile=True)
#     logger = logging.getLogger('base')
    
#     logger.info("===== Configuration =====")
#     logger.info(dict2str(vars(args)))
#     logger.info(f"Mode: {'Full Reference (GT Provided)' if has_gt else 'No-Reference (No GT)'}")
#     logger.info("==========================\n")

#     # Initialize IQA metrics
#     logger.info("Initializing IQA metrics...")
#     iqa_metrics = {}

#     # 1. Always initialize Non-Reference (NR) metrics
#     # Note: Ensure the metric names match what pyiqa expects or your custom logic
#     iqa_metrics['liqe'] = pyiqa.create_metric('liqe', device=device)
#     iqa_metrics['topiq_nr'] = pyiqa.create_metric('topiq_nr', device=device)
#     iqa_metrics['CLIPIQA'] = pyiqa.create_metric('clipiqa', device=device)
#     iqa_metrics['NIQE'] = pyiqa.create_metric('niqe', device=device)
#     iqa_metrics['MUSIQ'] = pyiqa.create_metric('musiq', device=device)
#     iqa_metrics['MANIQA'] = pyiqa.create_metric('maniqa', device=device)

#     # 2. Only initialize Reference metrics if GT is available
#     if has_gt:
#         logger.info("GT provided. Initializing PSNR, SSIM, LPIPS, DISTS, FID...")
#         iqa_metrics['PSNR'] = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
#         iqa_metrics['SSIM'] = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(device)
#         iqa_metrics['LPIPS'] = pyiqa.create_metric('lpips', device=device)
#         iqa_metrics['DISTS'] = pyiqa.create_metric('dists', device=device)
        
#         # Initialize FID separately
#         fid_metric = pyiqa.create_metric('fid', device=device)
#     else:
#         logger.info("No GT provided. Skipping Reference metrics (PSNR, SSIM, etc.) and FID.")

#     logger.info("IQA metrics initialized.\n")

#     logger.info("\n===== Starting Evaluation =====\n")

#     # Define which metrics are Non-Reference to handle call signatures
#     nr_metric_names = ['CLIPIQA', 'NIQE', 'MUSIQ', 'MANIQA', 'liqe', 'topiq_nr']

#     # Iterate over each directory pair
#     for dir_idx, init_dir in enumerate(args.inp_imgs):
#         gt_dir = args.gt_imgs[dir_idx] # This is None if not has_gt
        
#         img_sr_list = sorted(glob.glob(os.path.join(init_dir, '*.png')))
#         dir_name = os.path.basename(os.path.normpath(init_dir))

#         # Handle GT list
#         if has_gt and gt_dir:
#             img_gt_list = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
#             logger.info(f"Directory [{dir_name}]: {len(img_gt_list)} GT images vs {len(img_sr_list)} SR images.")
#             assert len(img_gt_list) == len(img_sr_list), f"Mismatch in number of images for directory: {dir_name}"
#         else:
#             img_gt_list = [None] * len(img_sr_list) # Placeholder list
#             logger.info(f"Directory [{dir_name}]: {len(img_sr_list)} SR images (No GT).")

#         # Initialize accumulators for average metrics
#         metrics_accum = {metric: 0.0 for metric in iqa_metrics.keys()}

#         logger.info(f"Testing Directory: [{dir_name}]")

#         # Iterate over each image pair
#         for img_idx, sr_path in enumerate(img_sr_list):
#             gt_path = img_gt_list[img_idx]
#             img_name = os.path.basename(sr_path)

#             start_time = time.time()

#             # Read and preprocess SR image
#             sr_img = cv2.imread(sr_path, cv2.IMREAD_COLOR)
#             if sr_img is None:
#                 logger.warning(f"Image read failed for {sr_path}. Skipping.")
#                 continue
            
#             # Read GT image only if needed
#             gt_img = None
#             gt_tensor = None
#             if has_gt and gt_path:
#                 gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
#                 if gt_img is None:
#                     logger.warning(f"GT Image read failed for {gt_path}. Skipping.")
#                     continue
#                 gt_tensor = img2tensor(gt_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device).contiguous() / 255.0

#             sr_tensor = img2tensor(sr_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device).contiguous() / 255.0

#             # Compute metrics
#             metrics = {}
#             with torch.no_grad():
#                 for name, metric in iqa_metrics.items():
#                     # Handle Reference vs No-Reference calls
#                     if name in nr_metric_names:
#                         metrics[name] = metric(sr_tensor).item()
#                     else:
#                         # Only compute reference metrics if GT is available
#                         if gt_tensor is not None:
#                             metrics[name] = metric(sr_tensor, gt_tensor).item()

#             # Accumulate metrics
#             for name in metrics:
#                 metrics_accum[name] += metrics[name]

#             # Calculate runtime
#             end_time = time.time()
#             runtime = end_time - start_time

#             # Log per-image metrics and runtime
#             metrics_str = "; ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
#             logger.info(f"{dir_name}/{img_name} | {metrics_str} | Runtime: {runtime:.2f} sec")

#         # Compute average metrics
#         num_images = len(img_sr_list)
#         if num_images > 0:
#             avg_metrics = {k: round(v / num_images, 4) for k, v in metrics_accum.items()}
#         else:
#             avg_metrics = {k: 0.0 for k in metrics_accum.keys()}

#         # Compute FID for the directory (Only if GT is provided)
#         fid_msg = ""
#         if has_gt and gt_dir:
#             try:
#                 fid_start_time = time.time()
#                 fid_value = fid_metric(gt_dir, init_dir).item()
#                 fid_end_time = time.time()
#                 fid_runtime = fid_end_time - fid_start_time
#                 fid_msg = f" | FID: {fid_value:.6f} | FID Runtime: {fid_runtime:.2f} sec"
#             except Exception as e:
#                 logger.error(f"FID computation failed: {e}")
#                 fid_msg = " | FID: Failed"
#         elif not has_gt:
#             fid_msg = " | FID: N/A (No GT)"

#         # Log average metrics for the directory
#         avg_metrics_str = "; ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
#         logger.info(f"\n===== Average Metrics for [{dir_name}] =====\n{avg_metrics_str}{fid_msg}\n")

#     logger.info("===== Evaluation Completed =====")

# if __name__ == "__main__":
#     main()



# Image Quality Assessment Script
# Evaluates metrics. Compatible with Reference (PSNR, SSIM, etc.) and No-Reference modes.
import pyiqa
# print(pyiqa.list_models()) # Optional: Print available models
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import glob
import argparse
import logging
from datetime import datetime
import time

import cv2
import numpy as np
import torch

from basicsr.utils import img2tensor

def get_timestamp():
    """Returns the current timestamp in a specific format."""
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    """Sets up a logger with specified configurations."""
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )
    logger.setLevel(level)

    if tofile:
        log_file = os.path.join(root, f"{phase}_{get_timestamp()}.log")
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

def dict2str(opt, indent=1):
    """Converts a dictionary to a formatted string for logging."""
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent * 2) + f"{k}:[\n"
            msg += dict2str(v, indent + 1)
            msg += ' ' * (indent * 2) + "]\n"
        else:
            msg += ' ' * (indent * 2) + f"{k}: {v}\n"
    return msg

def main():
    parser = argparse.ArgumentParser(description="Image Quality Assessment Script")

    parser.add_argument(
        "--inp_imgs",
        nargs="+",
        required=True,
        help="Path(s) to the input (SR) images directories."
    )

    # Modified: gt_imgs is now optional
    parser.add_argument(
        "--gt_imgs",
        nargs="+",
        required=False, 
        default=None,
        help="Path(s) to the ground truth (GT) images directories. Optional."
    )

    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Directory path to save the log files."
    )

    parser.add_argument(
        "--log_name",
        type=str,
        default='METRICS',
        help="Base name for the log files."
    )

    args = parser.parse_args()

    # Determine if GT is provided
    has_gt = args.gt_imgs is not None

    # Validate inputs if GT is provided
    if has_gt:
        if len(args.inp_imgs) != len(args.gt_imgs):
            print("Error: The number of input image directories and GT image directories must be the same.")
            sys.exit(1)
    else:
        # Create a dummy list of Nones for gt_imgs to simplify the loop later
        args.gt_imgs = [None] * len(args.inp_imgs)

    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create log directory if it doesn't exist
    os.makedirs(args.log, exist_ok=True)

    # Initialize logger
    try:
        args.log_name = args.inp_imgs[0].split('/')[-1] 
        if not args.log_name: # Handle trailing slash case
             args.log_name = args.inp_imgs[0].split('/')[-2]
    except IndexError:
        args.log_name = 'METRICS'
    
    phase_name = f'test_{args.log_name}' if has_gt else f'test_NR_{args.log_name}'
    setup_logger('base', args.log, phase_name, level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    
    logger.info("===== Configuration =====")
    logger.info(dict2str(vars(args)))
    logger.info(f"Mode: {'Full Reference (GT Provided)' if has_gt else 'No-Reference (No GT)'}")
    logger.info("==========================\n")

    # Initialize IQA metrics
    logger.info("Initializing IQA metrics...")
    iqa_metrics = {}

    # 1. Always initialize Non-Reference (NR) metrics
    # Note: Ensure the metric names match what pyiqa expects or your custom logic
    # Keys used here must match the 'nr_metric_names' list below EXACTLY.
    iqa_metrics['liqe'] = pyiqa.create_metric('liqe', device=device)
    iqa_metrics['topiq_nr'] = pyiqa.create_metric('topiq_nr', device=device)
    iqa_metrics['CLIPIQA'] = pyiqa.create_metric('clipiqa', device=device)
    iqa_metrics['NIQE'] = pyiqa.create_metric('niqe', device=device)
    iqa_metrics['MUSIQ'] = pyiqa.create_metric('musiq', device=device)
    iqa_metrics['MANIQA'] = pyiqa.create_metric('maniqa', device=device)

    # 2. Only initialize Reference metrics if GT is available
    if has_gt:
        logger.info("GT provided. Initializing PSNR, SSIM, LPIPS, DISTS, FID...")
        iqa_metrics['PSNR'] = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
        iqa_metrics['SSIM'] = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(device)
        iqa_metrics['LPIPS'] = pyiqa.create_metric('lpips', device=device)
        iqa_metrics['DISTS'] = pyiqa.create_metric('dists', device=device)
        
        # Initialize FID separately (it works on folders, not tensors usually in this context)
        fid_metric = pyiqa.create_metric('fid', device=device)
    else:
        logger.info("No GT provided. Skipping Reference metrics (PSNR, SSIM, etc.) and FID.")

    logger.info("IQA metrics initialized.\n")

    logger.info("\n===== Starting Evaluation =====\n")

    # === [CRITICAL FIX] ===
    # Explicitly define which keys in 'iqa_metrics' are No-Reference.
    # These metrics will ALWAYS receive only the SR image, ignoring GT even if it exists.
    nr_metric_names = ['CLIPIQA', 'NIQE', 'MUSIQ', 'MANIQA', 'liqe', 'topiq_nr']

    # Iterate over each directory pair
    for dir_idx, init_dir in enumerate(args.inp_imgs):
        gt_dir = args.gt_imgs[dir_idx] # This is None if not has_gt
        
        # Support multiple image extensions for SR
        sr_extensions = ['*.png', '*.jpg', '*.JPEG', '*.jpeg']
        img_sr_list = []
        for ext in sr_extensions:
            img_sr_list.extend(glob.glob(os.path.join(init_dir, ext)))
        img_sr_list = sorted(img_sr_list)
        
        dir_name = os.path.basename(os.path.normpath(init_dir))

        # Handle GT list
        if has_gt and gt_dir:
            # Support multiple image extensions for GT
            gt_extensions = ['*.png', '*.jpg', '*.JPEG', '*.jpeg']
            img_gt_list = []
            for ext in gt_extensions:
                img_gt_list.extend(glob.glob(os.path.join(gt_dir, ext)))
            img_gt_list = sorted(img_gt_list)

            logger.info(f"Directory [{dir_name}]: {len(img_gt_list)} GT images vs {len(img_sr_list)} SR images.")
            assert len(img_gt_list) == len(img_sr_list), f"Mismatch in number of images for directory: {dir_name}"
        else:
            img_gt_list = [None] * len(img_sr_list) # Placeholder list
            logger.info(f"Directory [{dir_name}]: {len(img_sr_list)} SR images (No GT).")

        # Initialize accumulators for average metrics
        metrics_accum = {metric: 0.0 for metric in iqa_metrics.keys()}

        logger.info(f"Testing Directory: [{dir_name}]")

        # Iterate over each image pair
        for img_idx, sr_path in enumerate(img_sr_list):
            gt_path = img_gt_list[img_idx]
            img_name = os.path.basename(sr_path)

            start_time = time.time()

            # Read and preprocess SR image
            sr_img = cv2.imread(sr_path, cv2.IMREAD_COLOR)
            if sr_img is None:
                logger.warning(f"Image read failed for {sr_path}. Skipping.")
                continue
            
            # Read GT image only if needed
            gt_img = None
            gt_tensor = None
            if has_gt and gt_path:
                gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
                if gt_img is None:
                    logger.warning(f"GT Image read failed for {gt_path}. Skipping.")
                    continue
                gt_tensor = img2tensor(gt_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device).contiguous() / 255.0

            sr_tensor = img2tensor(sr_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device).contiguous() / 255.0

            # Compute metrics
            metrics = {}
            with torch.no_grad():
                for name, metric in iqa_metrics.items():
                    # === [LOGIC FIX] ===
                    # Strictly enforce Single-Argument call for NR metrics
                    if name in nr_metric_names:
                        metrics[name] = metric(sr_tensor).item()
                    else:
                        # Only compute reference metrics if GT is available
                        if gt_tensor is not None:
                            metrics[name] = metric(sr_tensor, gt_tensor).item()

            # Accumulate metrics
            for name in metrics:
                metrics_accum[name] += metrics[name]

            # Calculate runtime
            end_time = time.time()
            runtime = end_time - start_time

            # Log per-image metrics and runtime
            metrics_str = "; ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            logger.info(f"{dir_name}/{img_name} | {metrics_str} | Runtime: {runtime:.2f} sec")

        # Compute average metrics
        num_images = len(img_sr_list)
        if num_images > 0:
            avg_metrics = {k: round(v / num_images, 4) for k, v in metrics_accum.items()}
        else:
            avg_metrics = {k: 0.0 for k in metrics_accum.keys()}

        # Compute FID for the directory (Only if GT is provided)
        fid_msg = ""
        if has_gt and gt_dir:
            try:
                fid_start_time = time.time()
                # FID requires paths, usually. pyiqa's fid metric handles paths.
                fid_value = fid_metric(gt_dir, init_dir).item()
                fid_end_time = time.time()
                fid_runtime = fid_end_time - fid_start_time
                fid_msg = f" | FID: {fid_value:.6f} | FID Runtime: {fid_runtime:.2f} sec"
            except Exception as e:
                logger.error(f"FID computation failed: {e}")
                fid_msg = " | FID: Failed"
        elif not has_gt:
            fid_msg = " | FID: N/A (No GT)"

        # Log average metrics for the directory
        avg_metrics_str = "; ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
        logger.info(f"\n===== Average Metrics for [{dir_name}] =====\n{avg_metrics_str}{fid_msg}\n")

    logger.info("===== Evaluation Completed =====")

if __name__ == "__main__":
    main()