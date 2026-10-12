"""Detection and multitask losses."""

from .multitask import MultiTaskLoss, consistency_residual, positive_u

__all__ = ["MultiTaskLoss", "consistency_residual", "positive_u"]
