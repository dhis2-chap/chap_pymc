from pydantic import BaseModel


class MCMCParams(BaseModel):
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.9
    random_seed: int = 42
    progressbar: bool = True
    n_iterations: int = 100000

    @classmethod
    def debug(self):
        return MCMCParams(draws=50, tune=10, chains=2, target_accept=0.8, random_seed=123, n_iterations=10)