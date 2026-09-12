"""Static invariants for the five LPI configurations; usable without ML libraries."""
from pathlib import Path

NAMES = ['LFM','NLFM','BPSK','BFSK','Frank']
AUGMENT_KEYS = ['hsv_h','hsv_s','hsv_v','degrees','translate','scale','shear','perspective','flipud','fliplr','bgr','mosaic','mixup','cutmix','copy_paste']


def validate_lpi_configs(root):
    from tools.validate_experiment_configs import resolved_config, normalized_value, nested_value
    root=Path(root)
    def check(cfg,expected):
        for key,value in expected.items():
            actual=normalized_value(nested_value(cfg,key))
            if actual!=value:raise ValueError(f'LPI {key}: expected={value!r}, actual={actual!r}')
    for i,model,indices in [(1,'repvit_m0_9.dist_450e_in1k',(0,1,2,3)),(2,'mobilenetv4_conv_small.e3600_r256_in1k',(1,2,3,4))]:
        cfg=resolved_config(root/f'configs/experiments/lpi/lpi-{i:03d}.yaml')
        batch = {1:16,2:24}[i]
        max_iter = 39600*200//batch
        period = 39600*5//batch
        check(cfg,{'EXPERIMENT.ID':f'lpi-{i:03d}','EXPERIMENT.DATASET':'lpi',
                   'MODEL.WEIGHTS':'','MODEL.TIMM.NAME':model,'MODEL.TIMM.PRETRAINED':True,'MODEL.TIMM.OUT_INDICES':indices,
                   'MODEL.FPN.OUT_CHANNELS':128,'MODEL.DiffusionDet.NUM_CLASSES':5,'MODEL.DiffusionDet.HIDDEN_DIM':128,
                   'MODEL.DiffusionDet.NUM_PROPOSALS':500,'MODEL.DiffusionDet.NUM_HEADS':6,'MODEL.DiffusionDet.HEAD_SHARING':'full','MODEL.DiffusionDet.SAMPLE_STEP':1,
                   'DATASETS.TRAIN':('lpi_train',),'DATASETS.TEST':('lpi_val',),'DATALOADER.FILTER_EMPTY_ANNOTATIONS':False,
                   'INPUT.MIN_SIZE_TRAIN':(640,),'INPUT.MAX_SIZE_TRAIN':640,'INPUT.MIN_SIZE_TEST':640,'INPUT.MAX_SIZE_TEST':640,
                   'INPUT.RANDOM_FLIP':'none','INPUT.CROP.ENABLED':False,'SEED':40244023,
                   'SOLVER.IMS_PER_BATCH':batch,'SOLVER.MAX_ITER':max_iter,'SOLVER.BASE_LR':{16:.0000224,24:.0000336}[batch],
                   'SOLVER.STEPS':(max_iter*7//10,max_iter*9//10),'SOLVER.WARMUP_ITERS':round(25000/batch),'SOLVER.AMP.ENABLED':False,
                   'SOLVER.CHECKPOINT_PERIOD':period,'SOLVER.CHECKPOINT_RETENTION':'latest',
                   'TEST.EVAL_PERIOD':period,'TEST.BEST_CHECKPOINT.ENABLED':True,'TEST.BEST_CHECKPOINT.METRIC':'bbox/AP'})
    expected_data={'path':str(root/'LPI_COCO'),'train':'images/train','val':'images/val','test':'images/test','names':dict(enumerate(NAMES))}
    if resolved_config(root/'configs/yolo/lpi/dataset.yaml')!=expected_data:
        raise ValueError('LPI YOLO 数据映射不一致')
    reference=None
    for i,model in enumerate(['yolo26n','yolo11n','yolov8n'],1):
        cfg=resolved_config(root/f'configs/yolo/lpi/yolo-lpi-{i:03d}.yaml')
        batch = {1:72,2:80,3:96}[i]
        check(cfg,{'model':model+'.pt','data':'configs/yolo/lpi/dataset.yaml','project':'runs/yolo/lpi',
                   'name':f'yolo-lpi-{i:03d}__{model}-pretrained-img640-bs{batch}-ep200-seed40244023',
                   'epochs':200,'batch':batch,'imgsz':640,'seed':40244023,'amp':False,'patience':0,
                   'single_cls':False,'pretrained':True,'augmentations':(),'multi_scale':0.,'close_mosaic':0,
                   'split':'val','max_det':100,'exist_ok':False,**{k:0. for k in AUGMENT_KEYS}})
        comparable={k:v for k,v in cfg.items() if k not in {'name','model','batch'}}
        if reference is not None and comparable!=reference:raise ValueError('YOLO LPI 配方不一致')
        reference=comparable
