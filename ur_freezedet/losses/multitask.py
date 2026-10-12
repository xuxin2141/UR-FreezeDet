import torch
import torch.nn.functional as F


def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def positive_u(raw, scale, floor):
    return torch.as_tensor(floor, device=raw.device, dtype=raw.dtype) + float(scale) * F.softplus(raw)


def consistency_residual(pi_hat, u_hat, umax):
    return torch.log1p(pi_hat) - torch.log(umax) + torch.log(u_hat)


class MultiTaskLoss:
    def __init__(self, weights=None, u_scale=1.0, log_pi_scale=1.0):
        self.weights = weights or {"det": 1, "phase": 1, "substage": 1, "u": .5, "pi": .5, "cons": .1}
        self.u_scale, self.log_pi_scale = max(float(u_scale), 1e-6), max(float(log_pi_scale), 1e-6)

    def __call__(self, outputs, batch, detection_loss=None, config=None):
        config = config or {"classification": "hierarchical", "use_u": True, "use_pi": True, "use_consistency": True}
        mask = batch["m_sub"].to(outputs["pi_hat"].device)
        phase = batch["phase"].to(outputs["phase_logit"].device)
        substage = batch["substage"].clamp_min(0).to(outputs["substage_logits"].device)
        u_target = batch["u_target"].to(outputs["u_hat"].device)
        pi_target = torch.nan_to_num(batch["pi_target"], nan=0.0).to(outputs["pi_hat"].device)
        umax = batch["umax"].to(outputs["u_hat"].device)
        terms = {}
        terms["det"] = outputs["u_hat"].sum() * 0 if detection_loss is None else detection_loss
        zero = outputs["u_hat"].sum() * 0
        terms["flat"] = F.cross_entropy(outputs["flat_logits"], batch["stage"].to(outputs["flat_logits"].device)) if config["classification"] == "flat" else zero
        terms["phase"] = F.binary_cross_entropy_with_logits(outputs["phase_logit"], phase) if config["classification"] == "hierarchical" else zero
        terms["substage"] = masked_mean(F.cross_entropy(outputs["substage_logits"], substage, reduction="none"), mask) if config["classification"] == "hierarchical" else zero
        terms["u"] = F.huber_loss((outputs["u_hat"] - u_target) / self.u_scale, torch.zeros_like(u_target), reduction="mean", delta=1.0)
        pi_error = (torch.log1p(outputs["pi_hat"]) - torch.log1p(pi_target)) / self.log_pi_scale
        terms["pi"] = masked_mean(F.huber_loss(pi_error, torch.zeros_like(pi_error), reduction="none", delta=1.0), mask)
        terms["cons"] = masked_mean(F.smooth_l1_loss(consistency_residual(outputs["pi_hat"], outputs["u_hat"], umax), torch.zeros_like(umax), reduction="none", beta=.1), mask)
        total = self.weights["det"] * terms["det"] + terms["flat"] + self.weights["phase"] * terms["phase"] + self.weights["substage"] * terms["substage"]
        if config.get("use_u"): total = total + self.weights["u"] * terms["u"]
        if config.get("use_pi"): total = total + self.weights["pi"] * terms["pi"]
        if config.get("use_consistency"): total = total + self.weights["cons"] * terms["cons"]
        terms["total"] = total
        return terms
