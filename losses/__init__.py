"""Loss registry: one import point for all criteria."""
from losses.alignment import EnergyLoss, ViewKLLoss
from losses.combined import CombinedAuxCriterion, CombinedHeadCriterion
from losses.detection import BoxLoss
from losses.jepa_losses import JEPALoss
from losses.metric import MetricLoss
from losses.prediction import DensePartLoss
from losses.reconstruction import ReconLoss, soft_dtw_divergence
from losses.sigreg import SIGReg

__all__ = ["JEPALoss", "SIGReg", "DensePartLoss", "ReconLoss", "BoxLoss",
           "MetricLoss", "ViewKLLoss", "EnergyLoss",
           "CombinedAuxCriterion", "CombinedHeadCriterion",
           "soft_dtw_divergence"]
