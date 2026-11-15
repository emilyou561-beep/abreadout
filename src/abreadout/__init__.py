"""abreadout: the checks and tests I run every time I read an A/B test."""

from abreadout.checks import SRMResult, srm_check
from abreadout.compare import BootstrapResult, Comparison, bootstrap_diff, holm, mean_test, proportion_test
from abreadout.cuped import CupedResult, cuped
from abreadout.power import minimum_detectable_effect, power, sample_size
from abreadout.readout import Readout, readout

__version__ = "0.1.0"

__all__ = [
    "BootstrapResult",
    "Comparison",
    "CupedResult",
    "Readout",
    "SRMResult",
    "bootstrap_diff",
    "cuped",
    "holm",
    "mean_test",
    "minimum_detectable_effect",
    "power",
    "proportion_test",
    "readout",
    "sample_size",
    "srm_check",
]
