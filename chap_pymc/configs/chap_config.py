import pydantic

from chap_pymc.curve_parametrizations.fourier_parametrization import FourierHyperparameters
from chap_pymc.inference_params import InferenceParams


class FullConfig(InferenceParams, FourierHyperparameters):
    ...

class ChapConfig(pydantic.BaseModel):
    user_options: FullConfig = FullConfig()
