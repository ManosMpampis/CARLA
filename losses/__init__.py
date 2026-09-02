from losses.classification_e2e import ClassificationLossE2E
from losses.pretext_loss import PretextLoss
from losses.tcl import TCLoss
from losses.losses import ClassificationLoss, ClassificationLossPart, ClassificationLossMoCo

__all__ = ['TCLoss', 'PretextLoss', 'ClassificationLossE2E', 'ClassificationLoss', 'ClassificationLossPart', 'ClassificationLossMoCo']