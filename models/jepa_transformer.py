"""Compatibility shim: transformer arm now lives in models.encoder."""
from models.encoder import TransformerEncoder1d

JEPATransformer = TransformerEncoder1d

__all__ = ["JEPATransformer", "TransformerEncoder1d"]
