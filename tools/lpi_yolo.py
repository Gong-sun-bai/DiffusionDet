"""LPI-specific Ultralytics dataset: immutable source, exact 640 validation, no silent omissions."""
from copy import copy
from pathlib import Path

from ultralytics.data.dataset import YOLODataset, DATASET_CACHE_VERSION
from ultralytics.data.utils import img2label_paths, get_hash, load_dataset_cache_file
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils.torch_utils import unwrap_model

from tools.lpi_common import ROOT, digest


class LpiYoloDataset(YOLODataset):
    def get_labels(self):
        self.label_files = img2label_paths(self.im_files)
        cache_dir = ROOT / '.cache/lpi_yolo_labels'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / (digest(self.label_files) + '.cache')
        try:
            cache = load_dataset_cache_file(cache_path)
            if cache['version'] != DATASET_CACHE_VERSION or cache['hash'] != get_hash(self.label_files + self.im_files):
                raise ValueError('cache mismatch')
        except (OSError, ValueError, KeyError, AttributeError, ModuleNotFoundError):
            cache = self.cache_labels(cache_path)
        labels = cache['labels']
        if len(labels) != len(self.im_files) or {x['im_file'] for x in labels} != set(self.im_files):
            raise ValueError('LPI 图片扫描存在缺失/损坏；禁止静默跳过')
        if cache['results'][1] or cache['results'][3]:
            raise ValueError('LPI 标签缺失或损坏')
        self.im_files = [x['im_file'] for x in labels]
        return labels


def make_dataset(args, img_path, batch, data, mode, stride):
    return LpiYoloDataset(img_path=img_path, imgsz=args.imgsz, batch_size=batch,
                          augment=mode == 'train', hyp=args, rect=False, cache=None,
                          single_cls=False, stride=stride, pad=0.0, prefix=f'{mode}: ',
                          task='detect', classes=None, data=data, fraction=1.0)


class LpiDetectionValidator(DetectionValidator):
    def build_dataset(self, img_path, mode='val', batch=None):
        return make_dataset(self.args, img_path, batch, self.data, mode, self.stride)


class LpiDetectionTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return make_dataset(self.args, img_path, batch, self.data, mode, stride)

    def get_validator(self):
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return LpiDetectionValidator(self.test_loader, save_dir=self.save_dir,
                                    args=copy(self.args), _callbacks=self.callbacks)
