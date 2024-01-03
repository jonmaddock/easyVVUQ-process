import easyvvuq as uq
import chaospy as cp
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

# Initialise cluster (describes a single node, or less if need less than that per worker)
# Don't need whole node in this case
# cluster = SLURMCluster(
#     cores=56,
#     processes=1,
#     memory="192GB",
#     account="UKAEA-AP001-CPU",
#     walltime="01:00:00",
#     queue="cclake",
# )

# Need less than a full node per worker
cluster = SLURMCluster(
    cores=1,
    processes=1,
    memory="4GB",
    account="UKAEA-AP001-CPU",
    walltime="00:10:00",
    queue="cclake",
)
cluster.scale(64)  # Number of workers
# print(cluster.job_script())

# Connect Dask client to remote cluster
client = Client(cluster)
# Code from now on submitted to batch queue

# Define campaign
WORK_DIR = "campaigns"
Path("campaigns").mkdir(exist_ok=True)
campaign = uq.Campaign(name="model_inputs", work_dir=WORK_DIR)

# Define parameter space
# Uncertainties from Alex's SA paper

params = {
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

# QoIs
# Violated constraint residuals
qois = [
    "vio_constr_res",
]

# Create encoder and decoder
encoder = uq.encoders.GenericEncoder(
    template_fname="demo_sol_no_f_IN.template", target_filename="IN.DAT"
)
decoder = uq.decoders.JSONDecoder(
    target_filename="constraints.json", output_columns=qois
)

cmd = "process -i IN.DAT"
actions = uq.actions.local_execute(encoder, cmd, decoder)

# Add the app
campaign.add_app(name="feasibility", params=params, actions=actions)

# %%
# Create PCE sampler
# Vary all 21 uncertain inputs
vary = {
    "fdene": cp.Uniform(
        1.1,
        1.3,
    ),
    "hfact": cp.Uniform(
        1.0,
        1.2,
    ),
    # "coreradius": cp.Uniform(
    #     0.45,
    #     0.75,
    # ),
    # "fimp_2": cp.Uniform(
    #     0.085,
    #     0.115,
    # ),
    "fimp_14": cp.Uniform(
        1.0e-5,
        1.0e-4,
    ),
    "psepbqarmax": cp.Uniform(
        8.7,
        9.7,
    ),
    # "flhthresh": cp.Uniform(
    #     0.85,
    #     1.15,
    # ),
    # "cboot": cp.Uniform(
    #     0.95,
    #     1.05,
    # ),
    # "peakfactrad": cp.Uniform(
    #     2.0,
    #     3.5,
    # ),
    "kappa": cp.Uniform(
        1.8,
        1.9,
    ),
    # "etaech": cp.Uniform(
    #     0.3,
    #     0.5,
    # ),
    # "feffcd": cp.Uniform(
    #     0.5,
    #     5.0,
    # ),
    # "etath": cp.Uniform(
    #     0.36,
    #     0.4,
    # ),
    # "etaiso": cp.Uniform(
    #     0.75,
    #     0.95,
    # ),
    # "boundl_18": cp.Uniform(
    #     3.25,
    #     3.75,
    # ),
    "pinjalw": cp.Uniform(
        51.0,
        61.0,
    ),
    "alstroh": cp.Uniform(
        6.0e8,
        7.2e8,
    ),
    # "sig_tf_wp_max": cp.Uniform(
    #     5.2e8,
    #     6.4e8,
    # ),
    # "aspect": cp.Uniform(
    #     3.0,
    #     3.2,
    # ),
    # "boundu_2": cp.Uniform(
    #     11.0,
    #     12.0,
    # ),
    "triang": cp.Uniform(
        0.4,
        0.6,
    ),
}
print("Generating PCE sampler...", flush=True)
pce_sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=3)
print(f"Number of samples = {pce_sampler.n_samples}", flush=True)

# Add pce_sampler to campaign
campaign.set_sampler(pce_sampler)

# Draw samples, execute and collate
campaign.execute(pool=client).collate(progress_bar=True)
samples = campaign.get_collation_result()

# Analyse a single output variable, vio_constr_res
results = campaign.analyse(qoi_cols=["vio_constr_res"])

# Get its distribution
dist = results.get_distribution(qoi="vio_constr_res")

# Locations for density function to be evaluated
# (This is taken from easyvvuq's fusion tutorial)
x = np.linspace(dist.lower[0], dist.upper[0])
pdf = dist.pdf(x)

# Plot PDF for `vio_constr_res`
ax = sns.lineplot(x=x, y=pdf, markers=True)
ax.set_title("Distribution for vio_constr_res")
ax.set_xlabel("vio_constr_res")
ax.set_ylabel("Probability density")

# Sobol indices
# results.plot_moments(qoi="vio_constr_res")
# results.plot_sobols_first("vio_constr_res") # only for vecotr qois. Like constraint vectors?

# Plot first-order Sobol indices, and save file.
fig, ax = plt.subplots()
results.plot_sobols_treemap(
    "vio_constr_res", figsize=(10, 10), ax=ax, filename="sobols"
)
# ax.set_title("Something")
