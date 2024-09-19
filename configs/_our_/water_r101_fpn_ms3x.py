_base_ = r'water_r50_fpn_ms3x.py/'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',checkpoint='torchvision://resnet101')
    )
)
data = dict(samples_per_gpu=1, workers_per_gpu=2)