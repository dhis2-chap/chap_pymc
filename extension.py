import uuid

import pymc as pm
from pytensor import tensor as pt


def continue_rw_process(idata, T_orig, T_extended, sigma_rw_val, L):
    """Continue Random Walk process from training end through historic period."""
    if T_extended <= T_orig:
        # No extension needed
        u_rw_last = (
            idata.posterior["u_rw"]
            .isel(u_rw_dim_0=-1)
            .mean(dim=["chain", "draw"])
            .values
        )
        return pt.as_tensor_variable(u_rw_last)

    # Get starting point from training
    u_rw_last = (
        idata.posterior["u_rw"]
        .isel(u_rw_dim_0=-1)
        .mean(dim=["chain", "draw"])
        .values
    )

    # Continue Random Walk for the extended period
    steps_to_continue = T_extended - T_orig
    u_current = pt.as_tensor_variable(u_rw_last)

    # Generate Random Walk steps for the historic extension period
    # Random walk: u_t = u_{t-1} + noise_t
    for step in range(steps_to_continue):
        unique_id = str(uuid.uuid4())[:8]
        noise = pm.Normal(f"historic_noise_{unique_id}_{step}", 0.0, sigma_rw_val, shape=(L,))
        u_current = u_current + noise

    return u_current
