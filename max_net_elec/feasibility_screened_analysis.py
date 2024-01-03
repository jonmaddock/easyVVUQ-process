# %% [markdown]
# # Screened feasibility analysis

# %%
import easyvvuq as uq
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pathlib as Path

# %%
# /// prefix is required before absolute path
db_location = "sqlite:////home/jon/code/notebooks/vvuq/feasibility/campaigns/model_inputsodum094c/campaign.db"
campaign = uq.Campaign(
    db_location=db_location, name="model_inputs", work_dir="campaigns"
)

samples = campaign.get_collation_result()
sample_count = samples.shape[0]
print(f"Number of samples = {sample_count}")


# %%
# Analyse a single output variable, vio_constr_res
results = campaign.analyse(qoi_cols=["vio_constr_res"])

# Get its distribution
dist = results.get_distribution(qoi="vio_constr_res")

# Locations for density function to be evaluated
# (This is taken from easyvvuq's fusion tutorial)
x = np.linspace(dist.lower[0], dist.upper[0])
pdf = dist.pdf(x)

# Plot
fig, ax = plt.subplots()
sns.lineplot(x=x, y=pdf, markers=True, ax=ax)
ax.set_title("Distribution for vio_constr_res")
ax.set_xlabel("vio_constr_res")
ax.set_ylabel("Probability density")
fig.savefig("vio_constr_res_dist.png")

# %% [markdown]
# ## Sobol indices

# %%
fig, ax = plt.subplots()
# results.plot_sobols_treemap("vio_constr_res", figsize=(10, 10), ax=ax)
results.plot_sobols_treemap("vio_constr_res", ax=ax)
# ax.set_title("blah")
fig.savefig("sobols_treemap.png")

# %% [markdown]
# ## Barplot of Sobol indices

# %%
# results.sobols_first()
# results.sobols_second()
# results.sobols_total()

sobols_first = results.sobols_first()["vio_constr_res"]
sobols_first_df = pd.DataFrame(sobols_first)
# sobols_first_df

fig, ax = plt.subplots()
sns.barplot(data=sobols_first_df, orient="h", ax=ax)
ax.set_title("First-order Sobol indices for violated constraint residuals")
ax.set_xlabel("First-order Sobol index")
ax.set_ylabel("Uncertain input")
fig.savefig("fo_sobols_bar.png")

# %%
sobols_total = results.sobols_total()["vio_constr_res"]

# Calculate higher-order Sobol indices
sobols_higher = {}
for key in sobols_first:
    sobols_higher[key] = sobols_total[key] - sobols_first[key]

# Combine first and higher-order into df
sobols_first_and_higher = {}
for key in sobols_first:
    sobols_first_and_higher[key] = [sobols_first[key][0], sobols_higher[key][0]]

sobols_first_and_higher_df = pd.DataFrame(
    sobols_first_and_higher, index=["First-order", "Higher-order"]
)
sobols_first_and_higher_df = sobols_first_and_higher_df.reset_index()
sobols_first_and_higher_df
sobols_first_and_higher_df_melted = sobols_first_and_higher_df.melt(id_vars="index")
sobols_first_and_higher_df_melted

# Plot
fig, ax = plt.subplots()
ax = sns.barplot(
    data=sobols_first_and_higher_df_melted,
    x="value",
    y="variable",
    hue="index",
    orient="h",
)
ax.set_title("First- and higher-order Sobol indices for violated constraint residuals")
ax.set_xlabel("Sobol index")
ax.set_ylabel("Uncertain input")
fig.savefig("fo_ho_sobols_bar.png")
