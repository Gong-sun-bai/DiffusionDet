"""Attach LPI input provenance after train_net's native directory guard, before training."""
import argparse
import hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train_net
from tools.lpi_common import read_json, write_json
from detectron2.engine import default_argument_parser, launch


def main():
    pre=argparse.ArgumentParser(add_help=False)
    pre.add_argument('--preflight',type=Path,required=True)
    known,rest=pre.parse_known_args()
    original_setup=train_net.setup
    native_trainer=train_net.Trainer
    class ProvenanceTrainer(native_trainer):
        @classmethod
        def build_model(cls,cfg):
            model=super().build_model(cfg)
            path=Path(cfg.OUTPUT_DIR)/'lpi_initialization.json'
            if not path.exists():
                h=hashlib.sha256()
                for name,tensor in sorted(model.backbone.state_dict().items()):
                    value=tensor.detach().cpu().contiguous()
                    h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
                write_json(path,{'backbone':cfg.MODEL.TIMM.NAME,'pretrained':cfg.MODEL.TIMM.PRETRAINED,
                                  'effective_backbone_state_sha256':h.hexdigest(),
                                  'parameter_count':sum(p.numel() for p in model.parameters()),
                                  'note':'模型构建后、恢复完整训练检查点前的有效 backbone 初始化指纹。'})
            return model
    train_net.Trainer=ProvenanceTrainer
    def setup(args):
        cfg,context,existing=original_setup(args)
        if existing is None:
            path=Path(context.run_dir)/'lpi_preflight.json'
            if not path.exists():
                from tools.lpi_source_snapshot import persist_preflight
                persist_preflight(known.preflight,path)
        return cfg,context,existing
    train_net.setup=setup
    # Native custom CLI is configured in train_net's __main__; our formal entry needs only resume.
    args=default_argument_parser().parse_args(rest)
    args.run_until_iter=None
    train_net.main(args)

if __name__=='__main__':main()
