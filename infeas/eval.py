"""Common module for evaluating infeasibilities."""

import easyvvuq as uq
import chaospy as cp
from pathlib import Path
from dask.distributed import Client
from typing import Sequence, Dict, Any

# Dir for storing runs and results db
WORK_DIR = "campaigns"

# Define parameter space
# 21 Uncertainties from Alex's SA paper
PARAMS = {
    "fdene": {
        "type": "float",
        "min": 1.1,
        "max": 1.3,
        "default": 1.2,
    },  # check: not sure if this is right var. I think ok
    "hfact": {"type": "float", "min": 1.0, "max": 1.2, "default": 1.2},
    "coreradius": {"type": "float", "min": 0.45, "max": 0.75, "default": 0.75},  # ok
    "fimp_2": {"type": "float", "min": 0.085, "max": 0.115, "default": 0.1},  # ok
    "fimp_14": {
        "type": "float",
        "min": 1.0e-5,
        "max": 1.0e-4,
        "default": 1.0e-5,
    },  # ok
    "psepbqarmax": {"type": "float", "min": 8.7, "max": 9.7, "default": 9.0},  # ok
    "flhthresh": {"type": "float", "min": 0.85, "max": 1.15, "default": 1.15},  # ok
    "cboot": {
        "type": "float",
        "min": 0.95,
        "max": 1.05,
        "default": 1.0,
    },  # ok
    "peakfactrad": {"type": "float", "min": 2.0, "max": 3.5, "default": 3.33},  # ok
    "kappa": {"type": "float", "min": 1.8, "max": 1.9, "default": 1.848},  # ok
    "etaech": {"type": "float", "min": 0.3, "max": 0.5, "default": 0.4},  # ok
    "feffcd": {"type": "float", "min": 0.5, "max": 5.0, "default": 1.0},  # ok
    "etath": {"type": "float", "min": 0.36, "max": 0.4, "default": 0.375},  # ok
    "etaiso": {"type": "float", "min": 0.75, "max": 0.95, "default": 0.9},  # ok
    "boundl_18": {
        "type": "float",
        "min": 3.25,
        "max": 3.75,
        "default": 3.25,
    },  # q^95_min, ok
    "pinjalw": {"type": "float", "min": 51.0, "max": 61.0, "default": 61.0},  # ok
    "alstroh": {"type": "float", "min": 6.0e8, "max": 7.2e8, "default": 6.6e8},  # ok
    "sig_tf_wp_max": {
        "type": "float",
        "min": 5.2e8,
        "max": 6.4e8,
        "default": 6.4e8,
    },  # ok, but might need sig_tf_case_max to be the same too
    "aspect": {"type": "float", "min": 3.0, "max": 3.2, "default": 3.1},
    "boundu_2": {
        "type": "float",
        "min": 11.0,
        "max": 12.0,
        "default": 12.0,
    },  # B_T^max, ok
    "triang": {"type": "float", "min": 0.4, "max": 0.6, "default": 0.5},  # ok
    "vary_param": {
        "type": "string",
        "default": "",
    },  # param being changed: used for analysis only
}

# QoIs/responses
# Violated constraint residuals
QOIS = [
    "obj_func",
    "eq_1",
    "eq_2",
    "eq_11",
    "ineq_5",
    "ineq_8",
    "ineq_9",
    "ineq_13",
    "ineq_15",
    "ineq_30",
    "ineq_16",
    "ineq_24",
    "ineq_25",
    "ineq_26",
    "ineq_27",
    "ineq_33",
    "ineq_34",
    "ineq_35",
    "ineq_36",
    "ineq_60",
    "ineq_62",
    "ineq_65",
    "ineq_72",
    "ineq_79",
    "ineq_81",
    "ineq_68",
    "ineq_31",
    "ineq_32",
    "rms_vio_constr_res",
]


def vary_params(params: Sequence) -> Dict:
    """Create distributions for the parameters to be varied.

    Uses the bounds for each parameter in the PARAMS dict to define the lower
    and upper bound on the uniform distribution.

    Parameters
    ----------
    params : Sequence
        Parameter names to vary

    Returns
    -------
    Dict
        Distributions for each parameter
    """
    vary = {}
    # For each param name, find attributes in PARAMS to define bounds on
    # distribution
    for param in params:
        attrs = PARAMS[param]
        vary[param] = cp.Uniform(attrs["min"], attrs["max"])

    return vary


def evaluate(
    cluster: Any,
    vary_param_names: Sequence,
    campaign_name: str = "campaign",
    template_fname: str = None,
    polynomial_order: int = 3,
):
    """Evaluate samples for parameters according to the PCE method.

    Parameters
    ----------
    cluster : Any
        Definition of each Dask worker
    vary_param_names : Sequence
        Names of parameters to vary
    campaign_name : str, optional
        Name of the study, by default "campaign"
    template_fname : str, optional
        Filename of input template to insert values into, by default None
    polynomial_order : int, optional
        PCE order, by default 3
    """
    # Connect Dask client to remote cluster
    client = Client(cluster)
    # Code from now on submitted to batch queue

    # Define campaign
    Path(WORK_DIR).mkdir(exist_ok=True)

    campaign = uq.Campaign(name=campaign_name, work_dir=WORK_DIR)

    # Create encoder and decoder
    encoder = uq.encoders.GenericEncoder(
        template_fname=template_fname, target_filename="IN.DAT"
    )
    # TODO Possibly change to "responses"
    decoder = uq.decoders.JSONDecoder(target_filename="qois.json", output_columns=QOIS)

    cmd = "process -i IN.DAT"
    actions = uq.actions.local_execute(encoder, cmd, decoder)

    # Add the app
    campaign.add_app(name="feasibility", params=PARAMS, actions=actions)

    vary = vary_params(vary_param_names)

    # Create PCE sampler
    print("Generating PCE sampler...", flush=True)
    pce_sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=polynomial_order)
    print(f"Number of samples = {pce_sampler.n_samples}", flush=True)

    # Add pce_sampler to campaign
    campaign.set_sampler(pce_sampler)

    # Draw samples, execute and collate
    campaign.execute(pool=client).collate(progress_bar=True)
