# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings
import numpy as np

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)

# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', help='test config file path',default='/home/liw324/code/WaterMask/configs/_our_/water_r50_fpn_1x.py')
    # parser.add_argument('--checkpoint', help='checkpoint file',default='/home/liw324/code/WaterMask/out/water_r50_fpn_1x/latest.pth')
    parser.add_argument('--checkpoint', help='checkpoint file',default='/home/liw324/code/WaterMask/out_SUIM/20241016_173450_water_r50_fpn_1x/epoch_12.pth')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics',default="WaterMask/out_SUIM/20241016_143049_water_r50_fpn_1x/eval")
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=1,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
        default="segm")
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved',
        default='/home/liw324/code/WaterMask/out/eval/water_r50_fpn_1x/seg_out')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args



args = parse_args()

assert args.out or args.eval or args.format_only or args.show \
    or args.show_dir, \
    ('Please specify at least one operation (save/eval/format/show the '
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"')

if args.eval and args.format_only:
    raise ValueError('--eval and --format_only cannot be both specified')

if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
    raise ValueError('The output file must be a pkl file.')

cfg = Config.fromfile(args.config)

# replace the ${key} with the value of cfg.key
cfg = replace_cfg_vals(cfg)

# update data root according to MMDET_DATASETS
update_data_root(cfg)

if args.cfg_options is not None:
    cfg.merge_from_dict(args.cfg_options)

cfg = compat_cfg(cfg)

# set multi-process settings
setup_multi_processes(cfg)

# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True

if 'pretrained' in cfg.model:
    cfg.model.pretrained = None
elif 'init_cfg' in cfg.model.backbone:
    cfg.model.backbone.init_cfg = None

if cfg.model.get('neck'):
    if isinstance(cfg.model.neck, list):
        for neck_cfg in cfg.model.neck:
            if neck_cfg.get('rfp_backbone'):
                if neck_cfg.rfp_backbone.get('pretrained'):
                    neck_cfg.rfp_backbone.pretrained = None
    elif cfg.model.neck.get('rfp_backbone'):
        if cfg.model.neck.rfp_backbone.get('pretrained'):
            cfg.model.neck.rfp_backbone.pretrained = None

if args.gpu_ids is not None:
    cfg.gpu_ids = args.gpu_ids[0:1]
    warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                    'Because we only support single GPU mode in '
                    'non-distributed testing. Use the first GPU '
                    'in `gpu_ids` now.')
else:
    cfg.gpu_ids = [args.gpu_id]
cfg.device = get_device()
# init distributed env first, since logger depends on the dist info.
if args.launcher == 'none':
    distributed = False
else:
    distributed = True
    init_dist(args.launcher, **cfg.dist_params)

test_dataloader_default_args = dict(
    samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

# in case the test dataset is concatenated
if isinstance(cfg.data.test, dict):
    cfg.data.test.test_mode = True
    if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(
            cfg.data.test.pipeline)
elif isinstance(cfg.data.test, list):
    for ds_cfg in cfg.data.test:
        ds_cfg.test_mode = True
    if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
        for ds_cfg in cfg.data.test:
            ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

test_loader_cfg = {
    **test_dataloader_default_args,
    **cfg.data.get('test_dataloader', {})
}

rank, _ = get_dist_info()
# allows not to create
if args.work_dir is not None and rank == 0:
    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

# build the dataloader
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(dataset, **test_loader_cfg)

# build the model and load checkpoint
cfg.model.train_cfg = None
model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
fp16_cfg = cfg.get('fp16', None)
if fp16_cfg is not None:
    wrap_fp16_model(model)
checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
if args.fuse_conv_bn:
    model = fuse_conv_bn(model)
# old versions did not save class info in checkpoints, this walkaround is
# for backward compatibility
if 'CLASSES' in checkpoint.get('meta', {}):
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    model.CLASSES = dataset.CLASSES

def mask_code_to_image(mask_code):
    '''accept a 1-7 h*w array and turn it into h*w*rgb'''
    rows, cols = mask_code.shape
    bin_array = np.zeros((rows, cols, 3), dtype=int)
    
    for i in range(rows):
        for j in range(cols):
            binary_str = np.binary_repr(mask_code[i, j], width=3)
            # print(mask_code[i, j],":",binary_str)
            bin_array[i, j] = [int(bit) for bit in binary_str]
            
    return 255*bin_array

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    

    categories_UIIS_to_SUIM=(0,6,5,2,3,1,4,7)
    categories_SUIM_to_UIIS=(0,5,3,4,6,2,1,7)
    
    HD_IOU=[]   # 001 1 in SUIM     
    WR_IOU=[]   # 011 3 in SUIM
    RO_IOU=[]   # 100 4 in SUIM
    RI_IOU=[]   # 101 5 in SUIM
    FV_IOU=[]   # 110 6 in SUIM
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)
        
        # mask visualize
        img_name=data['img_metas'][0].data[0][0]['ori_filename'][:-4]
        filepath=f"/home/liw324/code/WaterMask/out/eval/water_r50_fpn_1x/pred_mask"

        # 如果路径不存在，则创建路径
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        mask_array = np.zeros([ori_h, ori_w],dtype=int)

        for cate in range(len(result[0][1])):
            mask_code=np.array(result[0][1][cate])
            if not np.any(mask_code):
                continue
            mask_code=mask_code.sum(axis=0)
            mask_code=mask_code>0
            # masks.append(mask)
            mask_array[mask_code==True] = categories_UIIS_to_SUIM[cate+1] * mask_code[mask_code == True]
            
            mask_img = mask_code_to_image(categories_UIIS_to_SUIM[cate+1] * mask_code).astype(np.uint8)
            img = Image.fromarray(mask_img)
            img.save(f'{filepath}/{img_name}_{categories_UIIS_to_SUIM[cate+1]}.bmp')
            
        mask_img = mask_code_to_image(mask_array).astype(np.uint8)
        img = Image.fromarray(mask_img)
        img.save(f'{filepath}/{img_name}_mask.bmp')
        
        
    return results


model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                            args.show_score_thr)
print("done")