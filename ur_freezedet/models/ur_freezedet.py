import torch
from torch import nn
import torch.nn.functional as F
from types import SimpleNamespace
from pathlib import Path
import ultralytics
from ultralytics.nn.tasks import DetectionModel

from ur_freezedet.losses.multitask import positive_u
from ur_freezedet.models.gates import LSF, RCSF


def hierarchical_probabilities(phase_logit, substage_logits):
    p_sub = torch.sigmoid(phase_logit).unsqueeze(1)
    conditional = substage_logits.softmax(1)
    return torch.cat((p_sub * conditional, 1.0 - p_sub), dim=1)


class URFreezeDet(nn.Module):
    def __init__(self, config, pretrained=False, u_scale=1.0, u_floor=1e-6, pi_gate_median=0.0, pi_gate_iqr=1.0):
        super().__init__()
        self.config = dict(config)
        yaml_path = Path(ultralytics.__file__).resolve().parent / "cfg" / "models" / "11" / "yolo11.yaml"
        self.yolo = DetectionModel(str(yaml_path), nc=1, verbose=False)
        self.yolo.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        if pretrained:
            checkpoint = torch.load(str(pretrained) if isinstance(pretrained, (str, Path)) else pretrained, map_location="cpu")
            source = checkpoint["model"].float().state_dict()
            current = self.yolo.state_dict()
            current.update({k: v for k, v in source.items() if k in current and current[k].shape == v.shape})
            self.yolo.load_state_dict(current)
        self._pyramid = None
        self.yolo.model[-1].register_forward_pre_hook(self._capture_pyramid)
        self.adapters = nn.ModuleList([nn.Sequential(nn.Linear(c, 128), nn.SiLU()) for c in (64, 128, 256)])
        self.base_fusion = nn.Sequential(nn.Linear(384, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU())
        self.flat_head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 3))
        self.phase_head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
        self.substage_head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 2))
        self.u_head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
        self.pi_head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
        self.lsf, self.rcsf = LSF(128), RCSF(128, pi_gate_median, pi_gate_iqr)
        self.u_scale, self.u_floor = float(u_scale), float(u_floor)

    def _capture_pyramid(self, module, inputs):
        self._pyramid = tuple(inputs[0])

    def forward(self, image):
        detect = self.yolo(image)
        if self._pyramid is None or len(self._pyramid) != 3: raise RuntimeError("Detect hook did not capture P3/P4/P5")
        q = [adapter(F.adaptive_avg_pool2d(feature, 1).flatten(1)) for adapter, feature in zip(self.adapters, self._pyramid)]
        h_base = self.base_fusion(torch.cat(q, dim=1))
        pi_hat = F.softplus(self.pi_head(h_base).squeeze(1))
        gate_type = self.config.get("fusion", "uniform")
        if gate_type == "lsf": h_substage, weights = self.lsf(q, h_base)
        elif gate_type == "rcsf": h_substage, weights = self.rcsf(q, h_base, pi_hat)
        else:
            weights = torch.full((image.shape[0], 3), 1 / 3, device=image.device, dtype=image.dtype)
            h_substage = sum(q) / 3
        flat_logits = self.flat_head(h_base)
        phase_logit = self.phase_head(h_base).squeeze(1)
        substage_logits = self.substage_head(h_substage)
        return {
            "detect": detect, "pyramid": self._pyramid, "h_base": h_base,
            "flat_logits": flat_logits, "phase_logit": phase_logit,
            "substage_logits": substage_logits,
            "stage_probabilities": hierarchical_probabilities(phase_logit, substage_logits),
            "u_hat": positive_u(self.u_head(h_base).squeeze(1), self.u_scale, self.u_floor),
            "pi_hat": pi_hat, "gate_weights": weights,
        }
