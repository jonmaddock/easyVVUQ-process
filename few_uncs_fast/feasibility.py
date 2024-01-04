# %% [markdown]
# # Feasibility UQ
#
# Take the generic DEMO solution, turned into an input file. Remove f-values at iteration vars, and replace their equality constraints with inequalities. Run PROCESS once-through with uncertain inputs, and the QoI as the value of constraints, i.e. the feasibility.
#
# Dask is used to parallelise.
# %%
import easyvvuq as uq
import chaospy as cp
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

# %% [markdown]
# ## Campaign to capture feasibility
#
# Using the epistemic uncertain inputs for the entire code, capture the distribution of constraint residuals.
#
# To start with, make just 4 inputs uncertain.

# %%
# Init cluster (describes a single node, or less if need less than that per worker)
# cluster = SLURMCluster(
#     cores=56,
#     processes=4,  # check docs
#     memory="192GB",
#     account="UKAEA-AP001-CPU",
#     walltime="01:00:00",
#     queue="cclake",
# )

# Need less than a full node per worker
# Also, activate Singularity env and conda env on each worker node
cluster = SLURMCluster(
    cores=1,
    processes=1,
    memory="4GB",
    account="UKAEA-AP001-CPU",
    walltime="00:05:00",
    queue="cclake",
    # python="singularity exec ~/process.sif bash ~/activate_py.sh",
)
cluster.scale(4)  # 4 workers
# print(cluster.job_script())

# Connect Dask client to remote cluster
client = Client(cluster)
# Code from now on submitted to batch queue

# %% [markdown]
# Execute the same as before, just with a `client` argument in the actions.
# %%
# Define campaign
WORK_DIR = "campaigns"
Path("campaigns").mkdir(exist_ok=True)
campaign = uq.Campaign(name="model_inputs", work_dir=WORK_DIR)

# Define parameter space
# Uncertainties from Alex's SA paper

params = {
    "fgwped": {
        "type": "float",
        "min": 1.1,
        "max": 1.3,
        "default": 1.1,
    },  # check: not sure if this is right var
    "hfact": {"type": "float", "min": 1.0, "max": 1.2, "default": 1.1},
    "coreradius": {"type": "float", "min": 0.45, "max": 0.75, "default": 0.75},
    "fimp_2": {"type": "float", "min": 0.085, "max": 0.115, "default": 0.1},  # check
    "fimp_14": {
        "type": "float",
        "min": 1.0e-5,
        "max": 1.0e-4,
        "default": 5e-5,
    },  # check
    "psepbqarmax": {"type": "float", "min": 8.7, "max": 9.7, "default": 9.2},
    "flhthresh": {"type": "float", "min": 0.85, "max": 1.15, "default": 1.15},  # check
    "bscfmax": {"type": "float", "min": 0.95, "max": 1.05, "default": 0.99},
    "peakfactrad": {"type": "float", "min": 2.0, "max": 3.5, "default": 3.33},
    "kappa": {"type": "float", "min": 1.8, "max": 1.9, "default": 1.8},  # check default
    "etaech": {"type": "float", "min": 0.3, "max": 0.5, "default": 0.4},
    "feffcd": {"type": "float", "min": 0.5, "max": 5.0, "default": 1.0},
    "etath": {"type": "float", "min": 0.36, "max": 0.4, "default": 0.375},
    "etaiso": {"type": "float", "min": 0.75, "max": 0.95, "default": 0.9},
    "boundl_18": {
        "type": "float",
        "min": 3.25,
        "max": 3.75,
        "default": 3.5,
    },  # q^95_min
    "pinjalw": {"type": "float", "min": 51.0, "max": 61.0, "default": 51.0},
    "alstroh": {"type": "float", "min": 6.0e8, "max": 7.2e8, "default": 6.6e8},
    "sig_tf_wp_max": {
        "type": "float",
        "min": 5.2e8,
        "max": 6.4e8,
        "default": 5.8e8,
    },  # winding pack, or casing?
    "aspect": {"type": "float", "min": 3.0, "max": 3.2, "default": 3.1},
    "boundu_2": {
        "type": "float",
        "min": 11.0,
        "max": 12.0,
        "default": 11.5,
    },  # B_T^max: check default
    "triang": {"type": "float", "min": 0.4, "max": 0.6, "default": 0.5},
    "out_file": {"type": "string", "default": "out.csv"},
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

# Create PCE sampler, 4 uncertainties
vary = {
    "aspect": cp.Uniform(3.0, 3.2),
    "triang": cp.Uniform(0.4, 0.6),
    "psepbqarmax": cp.Uniform(8.7, 9.7),
    "hfact": cp.Uniform(1.0, 1.2),
}
pce_sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=3)

# Add pce_sampler to campaign
campaign.set_sampler(pce_sampler)

# Draw samples, execute and collate
campaign.execute(pool=client).collate(progress_bar=True)
samples = campaign.get_collation_result()

# %%
samples

# %% [markdown]
# ## Plot some samples
#
# The most basic analysis.

# %%
# Plot 2 vars from sample against each other
# Input epistemic uncertainty aspect against vio_constr_res
ax = sns.regplot(x=samples["aspect"], y=samples["vio_constr_res"])
ax.set_xlabel("aspect")
ax.set_ylabel("vio_constr_res")
ax.set_title("Variability in vio_constr_res against aspect ratio")

# %% [markdown]
# ## Analysis
#
# Analyse vio_constr_residuals, the violated constraint residuals.
#
# ### KDE for `vio_constr_res` from EasyVVUQ (plotted with seaborn)

# %%
# Try to analyse all outputs variables: produces a linalg error
# results = campaign.analyse(qoi_cols=palph2_inputs)

# Analyse a single output variable, vio_constr_res
results = campaign.analyse(qoi_cols=["vio_constr_res"])

# Get its distribution
dist = results.get_distribution(qoi="vio_constr_res")

# Locations for density function to be evaluated
# (This is taken from easyvvuq's fusion tutorial)
x = np.linspace(dist.lower[0], dist.upper[0])
pdf = dist.pdf(x)

# Plot
ax = sns.lineplot(x=x, y=pdf, markers=True)
ax.set_title("Distribution for vio_constr_res")
ax.set_xlabel("vio_constr_res")
ax.set_ylabel("Probability density")
# %% [markdown]
# PDF for `vio_constr_res`. Appears right from looking at the EasyVVUQ tutorial (fusion Dask).

# %% [markdown]
# ## Sobol indices
# %%
# results.plot_moments(qoi="vio_constr_res")
# results.plot_sobols_first("vio_constr_res") # only for vecotr qois. Like constraint vectors?

fig, ax = plt.subplots()
results.plot_sobols_treemap(
    "vio_constr_res", figsize=(10, 10), ax=ax, filename="sobols"
)
