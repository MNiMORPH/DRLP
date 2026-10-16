"""
DRLP: Detachment-Limited River long Profile.

A MNiMORPH long-profile model, sibling to GRLP (gravel-bed) and SRLP
(sand-bed).  GRLP and SRLP are transport-limited and *diffusive*; DRLP is
detachment-limited and *advective*.  Erosion is set locally by the flow rather
than by a divergence of sediment flux, so a perturbation travels upstream as a
step instead of spreading and flattening.

    dz/dt = U - K Q**m S**n,      S = -dz/dx  (positive downstream)

SI units throughout; time in seconds.

The name states the assumption the model actually makes.  The stream-power law
is a parameterisation of detachment-limited incision and does not know whether
the bed is granite or shale, so the "bedrock" name is reserved for a future
mechanistic model (abrasion, plucking, tools and cover, block size).
"""

from ._version import __version__
from .incision import ConvergenceError, step
from .longprofile import LongProfile

__all__ = ["ConvergenceError", "LongProfile", "step", "__version__"]
