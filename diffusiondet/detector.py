# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import random
from pathlib import Path
from typing import List
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import batched_nms
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess

from detectron2.structures import Boxes, ImageList, Instances

from .loss import SetCriterionDynamicK, HungarianMatcherDynamicK
from .head import DynamicHead
from .util.box_ops import (
    aligned_generalized_box_iou,
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
)
from .util.misc import nested_tensor_from_tensor_list

__all__ = ["DiffusionDet"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


@META_ARCH_REGISTRY.register()
class DiffusionDet(nn.Module):
    """
    Implement DiffusionDet
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.DiffusionDet.NUM_CLASSES
        self.num_proposals = cfg.MODEL.DiffusionDet.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        self.num_heads = cfg.MODEL.DiffusionDet.NUM_HEADS
        inference_head_stage = int(cfg.MODEL.DiffusionDet.INFERENCE_HEAD_STAGE)
        if inference_head_stage == -1:
            self.inference_head_index = -1
        elif 1 <= inference_head_stage <= self.num_heads:
            self.inference_head_index = inference_head_stage - 1
        else:
            raise ValueError(
                "MODEL.DiffusionDet.INFERENCE_HEAD_STAGE must be -1 or in "
                f"[1, {self.num_heads}], got {inference_head_stage}"
            )

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility

        # build diffusion
        timesteps = 1000
        sampling_timesteps = cfg.MODEL.DiffusionDet.SAMPLE_STEP
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = float(cfg.MODEL.DiffusionDet.DDIM_ETA)
        self.self_condition = False
        self.scale = cfg.MODEL.DiffusionDet.SNR_SCALE
        self.box_renewal = bool(cfg.MODEL.DiffusionDet.BOX_RENEWAL)
        self.renewal_threshold = float(cfg.MODEL.DiffusionDet.RENEWAL_THRESHOLD)
        self.ensemble_mode = str(cfg.MODEL.DiffusionDet.ENSEMBLE_MODE)
        if self.ensemble_mode not in {
            "legacy_nonfinal",
            "final_only",
            "all_steps",
        }:
            raise ValueError(
                "MODEL.DiffusionDet.ENSEMBLE_MODE must be legacy_nonfinal, "
                f"final_only, or all_steps; got {self.ensemble_mode!r}"
            )
        self.use_ensemble = self.ensemble_mode in {
            "legacy_nonfinal",
            "all_steps",
        }

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        # Loss parameters:
        class_weight = cfg.MODEL.DiffusionDet.CLASS_WEIGHT
        giou_weight = cfg.MODEL.DiffusionDet.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DiffusionDet.L1_WEIGHT
        no_object_weight = cfg.MODEL.DiffusionDet.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.DiffusionDet.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        self.use_nms = cfg.MODEL.DiffusionDet.USE_NMS

        distillation = cfg.MODEL.DiffusionDet.DISTILLATION
        self.distillation_enabled = bool(distillation.ENABLED)
        self.distillation_teacher_weights = str(distillation.TEACHER_WEIGHTS)
        self.distillation_confidence_threshold = float(
            distillation.CONFIDENCE_THRESHOLD
        )
        self.distillation_temperature = float(distillation.TEMPERATURE)
        self.distillation_cls_weight = float(distillation.CLS_WEIGHT)
        self.distillation_l1_weight = float(distillation.L1_WEIGHT)
        self.distillation_giou_weight = float(distillation.GIOU_WEIGHT)
        self.__dict__["_distillation_teacher"] = None
        self.__dict__["_distillation_teacher_cfg"] = (
            self._teacher_cfg(cfg) if self.distillation_enabled else None
        )

        # Build Criterion.
        matcher = HungarianMatcherDynamicK(
            cfg=cfg, cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight, use_focal=self.use_focal
        )
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = SetCriterionDynamicK(
            cfg=cfg, num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight,
            losses=losses, use_focal=self.use_focal,)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    @staticmethod
    def _teacher_cfg(cfg):
        teacher_cfg = cfg.clone()
        teacher_cfg.defrost()
        teacher_cfg.MODEL.WEIGHTS = ""
        teacher_cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
        teacher_cfg.MODEL.RESNETS.DEPTH = 18
        teacher_cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
        teacher_cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
        teacher_cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        teacher_cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
        teacher_cfg.MODEL.FPN.OUT_CHANNELS = 128
        teacher_cfg.MODEL.DiffusionDet.HIDDEN_DIM = 128
        teacher_cfg.MODEL.DiffusionDet.NUM_HEADS = 6
        teacher_cfg.MODEL.DiffusionDet.HEAD_SHARING = "none"
        teacher_cfg.MODEL.DiffusionDet.INFERENCE_HEAD_STAGE = -1
        teacher_cfg.MODEL.DiffusionDet.DISTILLATION.ENABLED = False
        teacher_cfg.freeze()
        return teacher_cfg

    def _get_distillation_teacher(self):
        teacher = self.__dict__.get("_distillation_teacher")
        if teacher is not None:
            return teacher
        weights = Path(self.distillation_teacher_weights).expanduser()
        if not weights.is_file():
            raise FileNotFoundError(
                f"distillation teacher checkpoint does not exist: {weights}"
            )
        from detectron2.checkpoint import DetectionCheckpointer

        teacher = DiffusionDet(self.__dict__["_distillation_teacher_cfg"])
        DetectionCheckpointer(teacher).resume_or_load(str(weights.resolve()), resume=False)
        teacher.eval()
        teacher.requires_grad_(False)
        self.__dict__["_distillation_teacher"] = teacher
        return teacher

    def _distillation_losses(
        self,
        student_class,
        student_boxes,
        teacher_class,
        teacher_boxes,
        images_whwh,
    ):
        teacher_prob = torch.sigmoid(teacher_class)
        confidence = teacher_prob.amax(dim=-1)
        mask = confidence >= self.distillation_confidence_threshold
        if not bool(mask.any()):
            zero = student_class.sum() * 0.0
            return {
                "loss_kd_cls": zero,
                "loss_kd_bbox": zero,
                "loss_kd_giou": zero,
            }

        temperature = self.distillation_temperature
        cls_loss = F.binary_cross_entropy_with_logits(
            student_class[mask] / temperature,
            teacher_prob[mask],
        ) * (temperature ** 2)
        scale = images_whwh[:, None, :].expand_as(student_boxes)
        student_normalized = student_boxes / scale
        teacher_normalized = teacher_boxes / scale
        student_selected = student_normalized[mask]
        teacher_selected = teacher_normalized[mask]
        l1_loss = F.l1_loss(student_selected, teacher_selected)
        giou_loss = 1.0 - aligned_generalized_box_iou(
            student_selected, teacher_selected
        ).mean()
        return {
            "loss_kd_cls": cls_loss * self.distillation_cls_weight,
            "loss_kd_bbox": l1_loss * self.distillation_l1_weight,
            "loss_kd_giou": giou_loss * self.distillation_giou_weight,
        }

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, backbone_feats, images_whwh, x, t, x_self_cond=None, clip_x_start=False):
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = box_cxcywh_to_xyxy(x_boxes)
        x_boxes = x_boxes * images_whwh[:, None, :]
        outputs_class, outputs_coord = self.head(backbone_feats, x_boxes, t, None)

        if not self.training and self.inference_head_index != -1:
            outputs_class = outputs_class[self.inference_head_index][None]
            outputs_coord = outputs_coord[self.inference_head_index][None]

        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[:, None, :]
        x_start = box_xyxy_to_cxcywh(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord

    @torch.no_grad()
    def ddim_sample(self, batched_inputs, backbone_feats, images_whwh, images, clip_denoised=True, do_postprocess=True):
        batch = images_whwh.shape[0]
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=self.device)

        ensemble_score = [[] for _ in range(batch)]
        ensemble_label = [[] for _ in range(batch)]
        ensemble_coord = [[] for _ in range(batch)]
        final_outputs_class = None
        final_outputs_coord = None
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, outputs_class, outputs_coord = self.model_predictions(backbone_feats, images_whwh, img, time_cond,
                                                                         self_cond, clip_x_start=clip_denoised)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start
            final_outputs_class = outputs_class
            final_outputs_coord = outputs_coord

            collect_step = (
                self.sampling_timesteps > 1
                and (
                    self.ensemble_mode == "all_steps"
                    or (
                        self.ensemble_mode == "legacy_nonfinal"
                        and time_next >= 0
                    )
                )
            )
            if collect_step:
                raw_predictions = self.inference(
                    outputs_class[-1],
                    outputs_coord[-1],
                    images.image_sizes,
                    return_raw=True,
                )
                for image_index, (boxes, scores, labels) in enumerate(raw_predictions):
                    ensemble_coord[image_index].append(boxes)
                    ensemble_score[image_index].append(scores)
                    ensemble_label[image_index].append(labels)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            if self.box_renewal:
                next_images = []
                scores = torch.sigmoid(outputs_class[-1]).amax(dim=-1)
                for image_index in range(batch):
                    keep = scores[image_index] > self.renewal_threshold
                    kept_noise = pred_noise[image_index][keep]
                    kept_start = x_start[image_index][keep]
                    kept_count = int(keep.sum().item())
                    noise = torch.randn_like(kept_start)
                    updated = (
                        kept_start * alpha_next.sqrt()
                        + c * kept_noise
                        + sigma * noise
                    )
                    replenish = torch.randn(
                        self.num_proposals - kept_count,
                        4,
                        device=img.device,
                        dtype=img.dtype,
                    )
                    next_images.append(torch.cat((updated, replenish), dim=0))
                img = torch.stack(next_images)
            else:
                noise = torch.randn_like(img)
                img = (
                    x_start * alpha_next.sqrt()
                    + c * pred_noise
                    + sigma * noise
                )

        if final_outputs_class is None or final_outputs_coord is None:
            raise RuntimeError("DDIM sampling produced no predictions")

        if self.use_ensemble and self.sampling_timesteps > 1:
            results = []
            for image_index, image_size in enumerate(images.image_sizes):
                if not ensemble_coord[image_index]:
                    raise RuntimeError(
                        f"ensemble mode {self.ensemble_mode!r} collected no predictions"
                    )
                box_pred_per_image = torch.cat(ensemble_coord[image_index], dim=0)
                scores_per_image = torch.cat(ensemble_score[image_index], dim=0)
                labels_per_image = torch.cat(ensemble_label[image_index], dim=0)
                if self.use_nms:
                    keep = batched_nms(
                        box_pred_per_image,
                        scores_per_image,
                        labels_per_image,
                        0.5,
                    )
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]

                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)
        else:
            output = {
                'pred_logits': final_outputs_class[-1],
                'pred_boxes': final_outputs_coord[-1],
            }
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        return results

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prepare Proposals.
        if not self.training:
            results = self.ddim_sample(batched_inputs, features, images_whwh, images)
            return results

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets, x_boxes, noises, t = self.prepare_targets(gt_instances)
            t = t.squeeze(-1)
            x_boxes = x_boxes * images_whwh[:, None, :]

            outputs_class, outputs_coord = self.head(features, x_boxes, t, None)
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            if self.distillation_enabled:
                teacher = self._get_distillation_teacher()
                with torch.no_grad():
                    teacher_src = teacher.backbone(images.tensor)
                    teacher_features = [
                        teacher_src[name] for name in teacher.in_features
                    ]
                    teacher_class, teacher_coord = teacher.head(
                        teacher_features,
                        x_boxes,
                        t,
                        None,
                    )
                loss_dict.update(
                    self._distillation_losses(
                        outputs_class[-1],
                        outputs_coord[-1],
                        teacher_class[-1],
                        teacher_coord[-1],
                        images_whwh,
                    )
                )
            return loss_dict

    def prepare_diffusion_repeat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        num_repeat = self.num_proposals // num_gt  # number of repeat except the last gt box in one image
        repeat_tensor = [num_repeat] * (num_gt - self.num_proposals % num_gt) + [num_repeat + 1] * (
                self.num_proposals % num_gt)
        assert sum(repeat_tensor) == self.num_proposals
        random.shuffle(repeat_tensor)
        repeat_tensor = torch.tensor(repeat_tensor, device=self.device)

        gt_boxes = (gt_boxes * 2. - 1.) * self.scale
        x_start = torch.repeat_interleave(gt_boxes, repeat_tensor, dim=0)

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_diffusion_concat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 4,
                                          device=self.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
            box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        x_start = (x_start * 2. - 1.) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_targets(self, targets):
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes)
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts)

    def inference(self, box_cls, box_pred, image_sizes, return_raw=False):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        raw_results = []

        if self.use_focal or self.use_fed_loss:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                if return_raw:
                    raw_results.append(
                        (box_pred_per_image, scores_per_image, labels_per_image)
                    )
                    continue

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                if return_raw:
                    raw_results.append(
                        (box_pred_per_image, scores_per_image, labels_per_image)
                    )
                    continue

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return raw_results if return_raw else results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
