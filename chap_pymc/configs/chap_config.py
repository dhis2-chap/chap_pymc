import pydantic

from chap_pymc.curve_parametrizations.fourier_parametrization import FourierHyperparameters
from chap_pymc.inference_params import InferenceParams
from chap_pymc.transformations.model_input_creator import FourierInputCreator


class FullConfig(InferenceParams, FourierHyperparameters, FourierInputCreator.Params):
    ...

class ChapConfig(pydantic.BaseModel):
    user_options: FullConfig = FullConfig()
