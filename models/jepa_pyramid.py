"""Compatibility shim: old pyramid import path.

New blocks live in models.blocks / models.encoder / models.predictor.
Keeps configs, tests, and docs importing from here working.
"""
from models.blocks import CausalConv1d, ConvBlock1d, StackedConvBlock
from models.encoder import PyramidEncoder
from models.predictor import GRUPredictor, TCNPredictor

CausalTCNPredictor = TCNPredictor
PyramidLevel = StackedConvBlock

__all__ = ["PyramidEncoder", "PyramidLevel", "StackedConvBlock",
           "ConvBlock1d", "CausalConv1d", "CausalTCNPredictor",
           "TCNPredictor", "GRUPredictor"]


def cumulative_strides(chain):
    """Running product of a stride chain starting with the stem's own 1."""
    out, acc = [], 1
    for s in chain:
        acc *= s
        out.append(acc)
    return out
