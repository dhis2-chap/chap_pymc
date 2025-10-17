from typing import Literal
from pydantic import BaseModel


class InferenceParams(BaseModel):
    method: Literal['hmc', 'advi'] = 'hmc'

    # HMC/NUTS parameters
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.9

    # ADVI parameters
    n_iterations: int = 100000

    # Common parameters
    random_seed: int = 42
    progressbar: bool = True

    @classmethod
    def debug(cls):
        return InferenceParams(
            draws=50,
            tune=10,
            chains=2,
            target_accept=0.8,
            random_seed=123,
            n_iterations=10
        )
