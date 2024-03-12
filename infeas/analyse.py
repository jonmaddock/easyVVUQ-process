import easyvvuq as uq
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from infeas.eval import QOIS, WORK_DIR


def read_campaign(campaign_name: str) -> pd.DataFrame:
    """Read in evaluated campaign and return dataframe.

    Fetches the latest campaign with matching name.

    Parameters
    ----------
    campaign_name : str
        Name of campaign

    Returns
    -------
    uq.campaign.Campaign
        easyVVUQ campaign
    pd.DataFrame
        Evaluation results

    Raises
    ------
    FileNotFoundError
        No database found with this campaign_name
    """
    print("Reading in campaign database.")

    # Find latest modified database with matching campaign_name
    # e.g. campaign_name = "example_local" matches
    # ./campaigns/example_local6ewb9dvw/campaign.db (easyVVUQ applies hash)
    latest_mod_time = 0.0
    for db_path in Path(WORK_DIR).glob(f"{campaign_name}*/campaign.db"):
        mod_time = db_path.stat().st_mtime
        if mod_time > latest_mod_time:
            latest_mod_time = mod_time
            db_location = str(db_path.resolve())

    if latest_mod_time == 0.0:
        raise FileNotFoundError("No database found.")

    # /// prefix is required before absolute path
    db_location_prefixed = f"sqlite:///{db_location}"
    campaign = uq.Campaign(
        db_location=db_location_prefixed, name=campaign_name, work_dir=WORK_DIR
    )

    samples = campaign.get_collation_result()
    sample_count = samples.shape[0]
    print(f"Campaign read in. Number of samples = {sample_count}")

    # Drop strange multi-index of 0
    # TODO Commenting out required for reliability analysis work: may break
    # other studies
    # samples.columns = samples.columns.droplevel(1)
    return campaign, samples


def describe_qois(results: pd.DataFrame) -> pd.DataFrame:
    """Describe QOIs in results.

    Parameters
    ----------
    results : pd.DataFrame
        Results of evaluations

    Returns
    -------
    pd.DataFrame
        Described QOIs in results
    """
    return results[QOIS].describe()


def get_vio_means_filt(results: pd.DataFrame) -> pd.DataFrame:
    """Means for individual violated constraints.

    Parameters
    ----------
    results : pd.DataFrame
        Evaluated samples

    Returns
    -------
    pd.DataFrame
        Means of individual violated constraints
    """
    # Get absolute values of violated constraint residuals
    # Abs values before mean: be fair to eq constraints (either side of 0)
    vio_means_series = results[QOIS].abs().mean().sort_values(ascending=False)
    vio_means = pd.DataFrame(vio_means_series, columns=["mean"])
    vio_means["variable"] = vio_means.index
    vio_means.reset_index(drop=True)

    # Exclude objf and vio-constr_res
    filter_qois = ["objf", "vio_constr_res"]
    constrs_mask = ~vio_means["variable"].isin(filter_qois)
    vio_means_filt = vio_means[constrs_mask]
    return vio_means_filt


def get_top_vio_means(results: pd.DataFrame) -> pd.DataFrame:
    """Return top 3 individual violated constraint means.

    Parameters
    ----------
    results : pd.DataFrame
        Evaluated samples

    Returns
    -------
    pd.DataFrame
        Top 3 individual violated constraint means
    """
    vio_means_filt = get_vio_means_filt(results)
    # Take top 3 violated contraints (by mean)
    vio_means_filt = vio_means_filt.reset_index(drop=True)
    top_vio_means = vio_means_filt[0:3]["variable"].to_list()
    return top_vio_means


def plot_violated_constraints_mean(
    results: pd.DataFrame,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    """Plot mean violated constraints.

    Parameters
    ----------
    results : pd.DataFrame
        Results of evaluations

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
        fig and ax for further plot customisation
    """
    print("Plotting violated constraint means.")
    vio_means_filt = get_vio_means_filt(results)

    # Plot
    fig, ax = plt.subplots()
    sns.barplot(data=vio_means_filt, x="mean", y="variable", ax=ax)
    ax.set_title("Mean violated constraint residuals under uncertainty")
    ax.set_xlabel("Mean violated residual")
    ax.set_ylabel("Constraint")

    return fig, ax


def plot_violated_constraints_freq(
    results: pd.DataFrame,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    """Plot frequency of constraint violations.

    Parameters
    ----------
    results : pd.DataFrame
        Results of evaluations

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
        fig and ax for further plot customisation
    """
    print("Plotting constraint violation freqencies.")
    constr_tol = 1e-3
    vio_counts = {}
    for qoi in QOIS:
        # Count rows (samples) where each constraint is violated
        vio_count = results[results[qoi].abs() > constr_tol].shape[0]
        vio_counts[qoi] = vio_count

    vio_counts
    vio_counts_df = pd.DataFrame(data=vio_counts, index=[0])

    vio_counts_df_melt = vio_counts_df.melt()
    vio_counts_df_melt = vio_counts_df_melt.sort_values(by="value", ascending=False)
    vio_counts_df_melt.reset_index(drop=True)
    filter_qois = ["objf", "vio_constr_res"]
    vio_counts_df_melt_filt = vio_counts_df_melt[
        ~vio_counts_df_melt["variable"].isin(filter_qois)
    ]

    # Plot
    fig, ax = plt.subplots()
    sns.barplot(data=vio_counts_df_melt_filt, x="value", y="variable", ax=ax)
    ax.set_title("Constraint violation frequency under uncertainty")
    ax.set_xlabel("Violation frequency")
    ax.set_ylabel("Constraint")

    return fig, ax


def plot_violated_constraints_dist(
    campaign: uq.campaign.Campaign,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    """Plot distribution of violated constraint residuals.

    Parameters
    ----------
    campaign : uq.campaign.Campaign
        Campaign containing evaluations

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
        fig and ax for further plot customisation
    """
    print("Plotting distribution of violated constraint residuals.")
    # Analyse a single output variable, vio_constr_res
    results = campaign.analyse(qoi_cols=["vio_constr_res"])

    # Get its distribution
    dist = results.get_distribution(qoi="vio_constr_res")

    # Locations for density function to be evaluated
    # (This is taken from easyvvuq's fusion tutorial)
    x = np.linspace(dist.lower[0], dist.upper[0], num=500)
    pdf = dist.pdf(x)

    # Plot
    fig, ax = plt.subplots()
    sns.lineplot(x=x, y=pdf, markers=True, ax=ax)
    ax.set_title("Distribution for vio_constr_res")
    ax.set_xlabel("RMS of violated constraint residuals")
    ax.set_ylabel("Probability density")
    ax.set_xlim([0.0, None])
    return fig, ax


def plot_individual_violated_constraint_dist(
    campaign: uq.campaign.Campaign, results: pd.DataFrame
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    """Plot distribution of violated constraint residuals.

    Parameters
    ----------
    campaign : uq.campaign.Campaign
        Campaign containing evaluations
    results : pd.DataFrame
        Results of sample evaluations

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
        fig and ax for further plot customisation
    """

    print("Plotting top 3 violated constraint distributions")
    top_vio_means = get_top_vio_means()

    # Analyse a single output variable, vio_constr_res
    results = campaign.analyse(qoi_cols=top_vio_means)

    # Get the distributions
    dists = []
    dist_lowest = 0.0
    dist_highest = 0.0
    for qoi in top_vio_means:
        dist = results.get_distribution(qoi=qoi)
        if dist.lower[0] < dist_lowest:
            dist_lowest = dist.lower[0]
        if dist.upper[0] > dist_highest:
            dist_highest = dist.upper[0]
        dists.append(dist)

    # Locations for density function to be evaluated
    x = np.linspace(dist_lowest, dist_highest, num=500)

    # Flip x to make +ve
    pdfs = {"x": -x}
    for constr_name, dist in zip(top_vio_means, dists):
        pdfs[constr_name] = dist.pdf(x)

    top_vio_means_df = pd.DataFrame(pdfs)
    top_vio_means_df
    top_vio_means_df_melt = top_vio_means_df.melt(id_vars="x", value_vars=top_vio_means)
    top_vio_means_df_melt
    # Plot
    fig, ax = plt.subplots()
    sns.lineplot(
        data=top_vio_means_df_melt,
        x="x",
        y="value",
        hue="variable",
        markers=True,
        ax=ax,
    )
    ax.set_title("Distribution of top 3 violated constraints")
    ax.set_xlabel("Violated constraint value")
    ax.set_ylabel("Probability density")
    ax.set_xlim([0.0, None])
    return fig, ax


def plot_objective_func(
    campaign: uq.campaign.Campaign,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    """Plot distribution of objective function.

    Parameters
    ----------
    campaign : uq.campaign.Campaign
        Campaign containing evaluations

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
        fig and ax for further plot customisation
    """
    print("Plotting objective function distribution")
    # Analyse a single output variable, vio_constr_res
    results = campaign.analyse(qoi_cols=["objf"])

    # Get its distribution
    dist = results.get_distribution(qoi="objf")

    # Locations for density function to be evaluated
    x = np.linspace(dist.lower[0], dist.upper[0], num=500)
    pdf = dist.pdf(x)

    # Plot
    fig, ax = plt.subplots()
    sns.lineplot(x=x, y=pdf, markers=True, ax=ax)
    ax.set_title("Distribution for objf")
    ax.set_xlabel("objf")
    ax.set_ylabel("Probability density")
    return fig, ax


def plot_sobols_violated_constraints(
    campaign: uq.campaign.Campaign,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    """Plot Sobol indices for RMS violated constraints.

    Parameters
    ----------
    campaign : uq.campaign.Campaign
        Campaign containing evaluations

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
        fig and ax for further plot customisation
    """
    print("Plotting Sobols for violated constraint residuals.")
    results = campaign.analyse(qoi_cols=["vio_constr_res"])
    fig, ax = plt.subplots()
    results.plot_sobols_treemap("vio_constr_res", figsize=(10, 10), ax=ax)
    return fig, ax


def plot_sobols_individual_violated_constraints(
    campaign: uq.campaign.Campaign, results: pd.DataFrame
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    """Plot Sobol indices for individual violated constraints.

    Parameters
    ----------
    campaign : uq.campaign.Campaign
        Campaign containing evaluations
    """
    print("Plotting individual constraint Sobols")
    top_vio_means = get_top_vio_means(results)
    results = campaign.analyse(qoi_cols=top_vio_means)
    for constr_name in top_vio_means:
        fig, ax = plt.subplots()
        try:
            results.plot_sobols_treemap(constr_name, ax=ax)
            # TODO Any way to render multiple plots in notebook?
            fig.savefig(f"{constr_name}_sobols_treemap.png")
            print(f"Plotted {constr_name} Sobols.")
        except:
            print(f"Couldn't plot {constr_name} Sobols.")


def plot_sobols_violated_constraints_first_order(
    campaign: uq.campaign.Campaign,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    """Plot barplot of Sobols for violated constraint residuals.

    Parameters
    ----------
    campaign : uq.campaign.Campaign
        Campaign containing evaluations

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
        fig and ax for further plot customisation
    """
    print("Plotting barplot of Sobols for violated constraint residuals.")
    # results.sobols_first()
    # results.sobols_second()
    # results.sobols_total()

    results = campaign.analyse(qoi_cols=["vio_constr_res"])
    sobols_first = results.sobols_first()["vio_constr_res"]
    sobols_first_df = pd.DataFrame(sobols_first)
    sobols_first_df_melted = sobols_first_df.melt()
    sobols_first_df_melted

    fig, ax = plt.subplots()
    sns.barplot(data=sobols_first_df_melted, x="value", y="variable", ax=ax)
    ax.set_title("First-order Sobol indices for violated constraint residuals")
    ax.set_xlabel("First-order Sobol index")
    ax.set_ylabel("Uncertain input")

    return fig, ax


def plot_sobols_individual_violated_constraints_first_and_higher_order(
    campaign: uq.campaign.Campaign,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    """Plot barplot of Sobols.

    Parameters
    ----------
    campaign : uq.campaign.Campaign
        Campaign containing evaluations

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
        fig and ax for further plot customisation
    """
    print("Plotting first and higher-order Sobols for violated constraint residuals.")
    results = campaign.analyse(qoi_cols=["vio_constr_res"])
    sobols_first = results.sobols_first()["vio_constr_res"]
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
    ax.set_title(
        "First- and higher-order Sobol indices for violated constraint residuals"
    )
    ax.set_xlabel("Sobol index")
    ax.set_ylabel("Uncertain input")
    return fig, ax
