"""Compatibility shim: old import path for the LeWM facade.

New code lives in models.lewm (clean rewrite, same contract).
This module keeps Trainer/Scorer/configs working unchanged.
"""
from models.lewm import ANTI_COLLAPSE_REGISTRY, LeWMModel
from models.predictor import GRUPredictor, MaskedReconPredictor, TCNPredictor, build_predictor

CausalTCNPredictor = TCNPredictor
JEPAModel = LeWMModel

PREDICTOR_REGISTRY = {"masked": MaskedReconPredictor, "tcn": TCNPredictor, "gru": GRUPredictor}

__all__ = ["JEPAModel", "LeWMModel", "PREDICTOR_REGISTRY",
           "ANTI_COLLAPSE_REGISTRY", "build_predictor",
           "CausalTCNPredictor", "GRUPredictor"]
