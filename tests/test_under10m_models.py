import types
import unittest

import torch
from torch import nn

from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.modeling import build_model

from diffusiondet import add_diffusiondet_config, add_mobilenetv4_config
from diffusiondet.detector import DiffusionDet, ModelPrediction
from diffusiondet.timm_backbone import TimmBackbone
from diffusiondet.util.box_ops import aligned_generalized_box_iou


def model_cfg():
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_mobilenetv4_config(cfg)
    cfg.merge_from_file("configs/experiments/sar_ship/sar-006-fixed256.yaml")
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = ""
    return cfg


class SharedHeadTests(unittest.TestCase):
    def test_default_none_preserves_historical_six_head_layout(self):
        cfg = model_cfg()
        model = build_model(cfg)
        self.assertEqual(model.head.head_sharing, "none")
        self.assertEqual(len(model.head.head_series), 6)
        self.assertIsNot(
            model.head.head_series[0].inst_interact,
            model.head.head_series[1].inst_interact,
        )
        keys = model.state_dict()
        self.assertIn("head.head_series.0.self_attn.in_proj_weight", keys)
        self.assertIn("head.head_series.5.self_attn.in_proj_weight", keys)

    def test_full_sharing_registers_one_head_for_six_iterations(self):
        cfg = model_cfg()
        cfg.MODEL.DiffusionDet.HEAD_SHARING = "full"
        model = build_model(cfg)
        self.assertEqual(model.head.num_heads, 6)
        self.assertEqual(len(model.head.head_series), 1)
        self.assertLess(sum(p.numel() for p in model.parameters()), 20_000_000)

    def test_heavy_sharing_reuses_only_expensive_modules(self):
        cfg = model_cfg()
        cfg.MODEL.DiffusionDet.HEAD_SHARING = "heavy"
        model = build_model(cfg)
        first, second = model.head.head_series[:2]
        self.assertIs(first.inst_interact, second.inst_interact)
        self.assertIs(first.linear1, second.linear1)
        self.assertIs(first.linear2, second.linear2)
        self.assertIsNot(first.self_attn, second.self_attn)
        self.assertIsNot(first.bboxes_delta, second.bboxes_delta)


class TimmBackboneTests(unittest.TestCase):
    def test_mobilenetv4_four_level_output(self):
        cfg = model_cfg()
        cfg.MODEL.TIMM.NAME = "mobilenetv4_conv_small.e3600_r256_in1k"
        cfg.MODEL.TIMM.PRETRAINED = False
        cfg.MODEL.TIMM.OUT_INDICES = (1, 2, 3, 4)
        backbone = TimmBackbone(cfg)
        shapes = backbone.output_shape()
        self.assertEqual([shapes[key].stride for key in backbone.out_features], [4, 8, 16, 32])
        outputs = backbone(torch.zeros(1, 3, 64, 64))
        self.assertEqual(list(outputs), list(backbone.out_features))
        self.assertEqual([value.shape[-2:] for value in outputs.values()], [(16, 16), (8, 8), (4, 4), (2, 2)])

    def test_repvit_shared_head_stays_below_screening_limit(self):
        cfg = model_cfg()
        cfg.MODEL.BACKBONE.NAME = "build_timm_fpn_backbone"
        cfg.MODEL.FPN.IN_FEATURES = ["stage1", "stage2", "stage3", "stage4"]
        cfg.MODEL.TIMM.NAME = "repvit_m0_9.dist_450e_in1k"
        cfg.MODEL.TIMM.PRETRAINED = False
        cfg.MODEL.TIMM.OUT_INDICES = (0, 1, 2, 3)
        cfg.MODEL.DiffusionDet.HEAD_SHARING = "full"
        model = build_model(cfg)
        self.assertLess(sum(p.numel() for p in model.parameters()), 9_500_000)


class SamplerTests(unittest.TestCase):
    def make_sampler(self, mode="all_steps", steps=2):
        model = DiffusionDet.__new__(DiffusionDet)
        nn.Module.__init__(model)
        model.device = torch.device("cpu")
        model.num_proposals = 3
        model.num_timesteps = 10
        model.sampling_timesteps = steps
        model.ddim_sampling_eta = 0.0
        model.objective = "pred_x0"
        model.self_condition = False
        model.box_renewal = True
        model.renewal_threshold = 0.5
        model.ensemble_mode = mode
        model.use_ensemble = mode in {"legacy_nonfinal", "all_steps"}
        model.use_nms = False
        model.use_focal = True
        model.use_fed_loss = False
        model.num_classes = 1
        model.register_buffer("alphas_cumprod", torch.linspace(0.99, 0.1, 10))

        calls = []

        def predictions(_, backbone_feats, images_whwh, image, time, *args, **kwargs):
            calls.append(time.clone())
            batch = image.shape[0]
            x_start = torch.zeros_like(image)
            pred_noise = torch.zeros_like(image)
            logits = torch.full((1, batch, 3, 1), 10.0)
            boxes = torch.tensor([0.0, 0.0, 8.0, 8.0]).repeat(1, batch, 3, 1)
            return ModelPrediction(pred_noise, x_start), logits, boxes

        model.model_predictions = types.MethodType(predictions, model)
        return model, calls

    def test_all_steps_includes_final_prediction_for_batch_two(self):
        model, calls = self.make_sampler("all_steps", steps=2)
        images = types.SimpleNamespace(image_sizes=[(8, 8), (8, 8)])
        results = model.ddim_sample(
            [{"height": 8, "width": 8}, {"height": 8, "width": 8}],
            [],
            torch.tensor([[8.0, 8.0, 8.0, 8.0]]).repeat(2, 1),
            images,
            do_postprocess=False,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0]), 6)
        self.assertEqual(len(results[1]), 6)

    def test_final_only_returns_only_last_step(self):
        model, calls = self.make_sampler("final_only", steps=2)

        def predictions(_, backbone_feats, images_whwh, image, time, *args, **kwargs):
            calls.append((time.clone(), image.clone()))
            batch = image.shape[0]
            value = float(time[0])
            x_start = torch.zeros_like(image)
            logits = torch.full((1, batch, 3, 1), 10.0)
            box = torch.tensor([value, value, value + 1, value + 1])
            boxes = box.repeat(1, batch, 3, 1)
            return ModelPrediction(torch.zeros_like(image), x_start), logits, boxes

        model.model_predictions = types.MethodType(predictions, model)
        images = types.SimpleNamespace(image_sizes=[(8, 8), (8, 8)])
        results = model.ddim_sample(
            [{"height": 8, "width": 8}, {"height": 8, "width": 8}],
            [],
            torch.tensor([[8.0, 8.0, 8.0, 8.0]]).repeat(2, 1),
            images,
            do_postprocess=False,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual([len(result) for result in results], [3, 3])
        self.assertTrue(
            torch.equal(
                results[0].pred_boxes.tensor[:, 0],
                torch.full((3,), float(calls[-1][0][0])),
            )
        )

    def test_batch_two_renewal_keeps_each_images_own_confident_queries(self):
        model, calls = self.make_sampler("final_only", steps=2)

        def predictions(_, backbone_feats, images_whwh, image, time, *args, **kwargs):
            calls.append((time.clone(), image.clone()))
            batch = image.shape[0]
            logits = torch.tensor(
                [[[[10.0], [-10.0], [-10.0]], [[10.0], [10.0], [-10.0]]]]
            )
            boxes = torch.tensor([0.0, 0.0, 8.0, 8.0]).repeat(1, batch, 3, 1)
            return (
                ModelPrediction(torch.zeros_like(image), torch.zeros_like(image)),
                logits,
                boxes,
            )

        model.model_predictions = types.MethodType(predictions, model)
        images = types.SimpleNamespace(image_sizes=[(8, 8), (8, 8)])
        torch.manual_seed(123)
        model.ddim_sample(
            [{"height": 8, "width": 8}, {"height": 8, "width": 8}],
            [],
            torch.tensor([[8.0, 8.0, 8.0, 8.0]]).repeat(2, 1),
            images,
            do_postprocess=False,
        )
        second_input = calls[1][1]
        self.assertEqual(tuple(second_input.shape), (2, 3, 4))
        self.assertTrue(torch.equal(second_input[0, :1], torch.zeros(1, 4)))
        self.assertTrue(torch.equal(second_input[1, :2], torch.zeros(2, 4)))
        self.assertFalse(torch.equal(second_input[0, 1:], torch.zeros(2, 4)))
        self.assertFalse(torch.equal(second_input[1, 2:], torch.zeros(1, 4)))

    def test_one_step_legacy_matches_final_only(self):
        images = types.SimpleNamespace(image_sizes=[(8, 8), (8, 8)])
        inputs = [{"height": 8, "width": 8}, {"height": 8, "width": 8}]
        sizes = torch.tensor([[8.0, 8.0, 8.0, 8.0]]).repeat(2, 1)
        outputs = []
        for mode in ("legacy_nonfinal", "final_only"):
            model, _ = self.make_sampler(mode, steps=1)
            torch.manual_seed(456)
            outputs.append(
                model.ddim_sample(
                    inputs, [], sizes, images, do_postprocess=False
                )
            )
        for legacy, final in zip(*outputs):
            self.assertTrue(
                torch.equal(legacy.pred_boxes.tensor, final.pred_boxes.tensor)
            )
            self.assertTrue(torch.equal(legacy.scores, final.scores))


class DistillationTests(unittest.TestCase):
    def test_aligned_giou_is_linear_in_number_of_queries(self):
        boxes = torch.tensor([[0.0, 0.0, 8.0, 8.0]]).repeat(10_000, 1)
        values = aligned_generalized_box_iou(boxes, boxes)
        self.assertEqual(values.shape, (10_000,))
        self.assertTrue(torch.equal(values, torch.ones_like(values)))

    def test_distillation_loss_backpropagates_only_through_student(self):
        cfg = model_cfg()
        model = build_model(cfg)
        student_class = torch.zeros(1, 3, 1, requires_grad=True)
        student_boxes = torch.tensor(
            [[[0.0, 0.0, 8.0, 8.0]] * 3], requires_grad=True
        )
        teacher_class = torch.full((1, 3, 1), 10.0)
        teacher_boxes = torch.tensor([[[1.0, 1.0, 7.0, 7.0]] * 3])
        losses = model._distillation_losses(
            student_class,
            student_boxes,
            teacher_class,
            teacher_boxes,
            torch.tensor([[8.0, 8.0, 8.0, 8.0]]),
        )
        sum(losses.values()).backward()
        self.assertIsNotNone(student_class.grad)
        self.assertIsNotNone(student_boxes.grad)
        self.assertFalse(teacher_class.requires_grad)
        self.assertFalse(teacher_boxes.requires_grad)
        self.assertFalse(any(key.startswith("_distillation_teacher") for key in model.state_dict()))


if __name__ == "__main__":
    unittest.main()
