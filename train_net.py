# ==========================================
# Modified by Shoufa Chen
# ===========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
DiffusionDet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import itertools
import weakref
from pathlib import Path
from typing import Any, Dict, List, Set
import logging
from collections import OrderedDict

import torch
from fvcore.nn.precise_bn import get_bn_modules

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, create_ddp_model, \
    AMPTrainer, SimpleTrainer, hooks
from detectron2.evaluation import COCOEvaluator, LVISEvaluator, verify_results
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.modeling import build_model
from detectron2.utils.events import EventStorage

from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA, add_mobilenetv4_config
from diffusiondet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer
from diffusiondet.datasets import register_project_datasets
from diffusiondet.checkpointing import (
    LatestCheckpointer,
    PersistentBestCheckpointer,
)
from diffusiondet.training_visualization import (
    TrainingPlotHook,
    TrainingPlotWriter,
)
from experiment_manager import (
    configure_experiment,
    finish_experiment,
    initialize_experiment,
    validate_output_state,
)

class Trainer(DefaultTrainer):
    """ Extension of the Trainer class adapted to DiffusionDet. """

    SCREENING_CHECKPOINT_PERIOD = 1000

    def __init__(self, cfg, run_until_iter=None):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()  # call grandfather's `__init__` while avoid father's `__init()`
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        ########## EMA ############
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        kwargs.update(may_get_ema_checkpointer(cfg, model))
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
            # trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.full_max_iter = int(cfg.SOLVER.MAX_ITER)
        self.run_until_iter = (
            min(self.full_max_iter, int(run_until_iter))
            if run_until_iter is not None
            else None
        )
        self.max_iter = (
            self.run_until_iter
            if self.run_until_iter is not None
            else self.full_max_iter
        )
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        # setup EMA
        may_build_model_ema(cfg, model)
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if 'lvis' in dataset_name:
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        else:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DiffusionDetDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def ema_test(cls, cfg, model, evaluators=None):
        # model with ema weights
        logger = logging.getLogger("detectron2.trainer")
        if cfg.MODEL_EMA.ENABLED:
            logger.info("Run evaluation with EMA.")
            with apply_model_ema_and_restore(model):
                results = cls.test(cfg, model, evaluators=evaluators)
        else:
            results = cls.test(cfg, model, evaluators=evaluators)
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = DiffusionDetWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        if cfg.MODEL_EMA.ENABLED:
            cls.ema_test(cfg, model, evaluators)
        else:
            res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            EMAHook(self.cfg, self.model) if cfg.MODEL_EMA.ENABLED else None,  # EMA hook
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        checkpoint_period = self.checkpoint_period(
            cfg.SOLVER.CHECKPOINT_PERIOD,
            self.run_until_iter,
            self.full_max_iter,
        )
        if comm.is_main_process():
            if cfg.SOLVER.CHECKPOINT_RETENTION == "latest":
                ret.append(
                    LatestCheckpointer(
                        self.checkpointer,
                        checkpoint_period,
                    )
                )
            else:
                ret.append(
                    hooks.PeriodicCheckpointer(
                        self.checkpointer,
                        checkpoint_period,
                    )
                )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            if cfg.TEST.BEST_CHECKPOINT.ENABLED:
                ret.append(
                    PersistentBestCheckpointer(
                        cfg.TEST.EVAL_PERIOD,
                        self.checkpointer,
                        cfg.TEST.BEST_CHECKPOINT.METRIC,
                        mode=cfg.TEST.BEST_CHECKPOINT.MODE,
                    )
                )
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
            if cfg.TRAINING_PLOTS.ENABLED:
                ret.append(
                    TrainingPlotHook(
                        TrainingPlotWriter(
                            cfg.OUTPUT_DIR,
                            max_iter=self.max_iter,
                        ),
                        period=cfg.TRAINING_PLOTS.PERIOD,
                        eval_period=cfg.TEST.EVAL_PERIOD,
                    )
                )
        return ret

    @classmethod
    def checkpoint_period(cls, configured_period, run_until_iter, full_max_iter):
        """Use denser recovery checkpoints only for bounded screening runs."""
        configured_period = int(configured_period)
        if run_until_iter is not None and int(run_until_iter) < int(full_max_iter):
            return min(configured_period, cls.SCREENING_CHECKPOINT_PERIOD)
        return configured_period

    def evaluate_resumed_rung(self, iteration):
        """Evaluate an already-reached screening rung without another update.

        A process can be interrupted after the rung checkpoint is saved but
        before its final validation finishes.  Detectron2's regular no-op
        train loop increments the storage iteration once, which would record
        the recovered metric at ``rung + 1``.  Running the hooks explicitly at
        the requested absolute iteration preserves the exact-rung evidence and
        still exercises EvalHook, the best-checkpoint hook, and all writers.
        """
        iteration = int(iteration)
        self.iter = iteration
        self.start_iter = iteration
        self.max_iter = iteration
        with EventStorage(iteration) as self.storage:
            try:
                self.before_train()
            finally:
                self.after_train()
        return getattr(self, "_last_eval_results", None)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    repository_root = Path(__file__).resolve().parent
    register_project_datasets(repository_root)
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_mobilenetv4_config(cfg)  # 添加 MobileNetV4 配置
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    context = configure_experiment(cfg, args, repository_root)
    cfg.OUTPUT_DIR = str(context.output_dir)
    if context.mode == "evaluation" and "://" not in cfg.MODEL.WEIGHTS:
        weights = Path(cfg.MODEL.WEIGHTS).expanduser()
        if not weights.is_absolute():
            weights = repository_root / weights
        cfg.MODEL.WEIGHTS = str(weights.resolve())
    cfg.freeze()

    # Plain training is intentionally idempotent.  A repeated command reports
    # the canonical result directory and exits before logging, model building,
    # data loading, or any write to the historical run.
    existing_run_dir = validate_output_state(context)
    if existing_run_dir is not None:
        if comm.is_main_process():
            print(
                "检测到相同实验的已有结果目录，跳过重复训练："
                f"{existing_run_dir}",
                flush=True,
            )
            print(
                "如需继续未完成实验，请使用同一配置并追加 --resume。",
                flush=True,
            )
        comm.synchronize()
        return cfg, context, existing_run_dir

    if comm.is_main_process():
        initialize_experiment(context, cfg)
    comm.synchronize()
    try:
        default_setup(cfg, args)
    except BaseException as error:
        if comm.is_main_process():
            finish_experiment(context, error=error)
        raise
    return cfg, context, None


def main(args):
    context = None
    try:
        cfg, context, existing_run_dir = setup(args)

        if existing_run_dir is not None:
            return {
                "status": "skipped_existing",
                "run_dir": str(existing_run_dir),
            }

        if args.eval_only:
            model = Trainer.build_model(cfg)
            kwargs = may_get_ema_checkpointer(cfg, model)
            if cfg.MODEL_EMA.ENABLED:
                EMADetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(
                    cfg.MODEL.WEIGHTS, resume=False
                )
            else:
                DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(
                    cfg.MODEL.WEIGHTS, resume=False
                )
            results = Trainer.ema_test(cfg, model)
            if cfg.TEST.AUG.ENABLED:
                results.update(Trainer.test_with_TTA(cfg, model))
            if comm.is_main_process():
                verify_results(cfg, results)
                finish_experiment(context, results=results)
            return results

        trainer = Trainer(cfg, run_until_iter=context.run_until_iter)
        trainer.resume_or_load(resume=args.resume)
        if (
            args.resume
            and context.run_until_iter is not None
            and trainer.start_iter >= trainer.max_iter
        ):
            results = trainer.evaluate_resumed_rung(context.run_until_iter)
        else:
            results = trainer.train()
        if comm.is_main_process():
            paused = (
                context.run_until_iter is not None
                and context.run_until_iter < int(cfg.SOLVER.MAX_ITER)
            )
            finish_experiment(
                context,
                results=results,
                status="paused" if paused else "completed",
            )
        return results
    except BaseException as error:
        if context is not None and comm.is_main_process():
            finish_experiment(context, error=error)
        raise


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--run-until-iter",
        type=int,
        default=None,
        help=(
            "Stop normally after this absolute iteration while preserving the "
            "full SOLVER.MAX_ITER schedule for a later --resume."
        ),
    )
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
