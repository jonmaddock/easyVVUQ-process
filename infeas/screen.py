"""Functions for screening with the single-parameter method."""
import easyvvuq as uq
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
from typing import Tuple
from infeas.eval import PARAMS, QOIS


def evaluate(campaign_name: str, template_fname: str) -> pd.DataFrame:
    """Evaluate single-parameter method samples with Process.

    Parameters
    ----------
    campaign_name : str
        Name of campaign run
    template_fname : str
        Input file template filename

    Returns
    -------
    pd.DataFrame
        Results of evaluation at each sample
    """
    # Define campaign
    WORK_DIR = "campaigns"
    Path(WORK_DIR).mkdir(exist_ok=True)
    campaign = uq.Campaign(name=campaign_name, work_dir=WORK_DIR)

    # Create encoder and decoder
    encoder = uq.encoders.GenericEncoder(
        template_fname=template_fname, target_filename="IN.DAT"
    )
    decoder = uq.decoders.JSONDecoder(target_filename="qois.json", output_columns=QOIS)

    # Define local Process run
    cmd = "process -i IN.DAT"
    actions = uq.actions.local_execute(encoder, cmd, decoder)

    # Add the app
    campaign.add_app(name="feasibility", params=PARAMS, actions=actions)

    # Add sampler to campaign
    dataframe_sampler = single_param_sampler()
    campaign.set_sampler(dataframe_sampler)
    print(f"Number of samples = {dataframe_sampler.n_samples()}", flush=True)

    # Draw samples, execute and collate
    campaign.execute().collate(progress_bar=True)
    results = campaign.get_collation_result()
    return results


def single_param_sampler() -> uq.sampling.dataframe_sampler.DataFrameSampler:
    """Create a custom single-param sampler for easyVVUQ to use.

    Returns
    -------
    uq.sampling.dataframe_sampler.DataFrameSampler
        Single-param sampler for easyVVUQ
    """
    # Create a sampler for single-parameter sampling
    # Dict of params to be varied: ready for dataframe conversion
    vary = {param: [] for param in PARAMS}

    # Remove vary_param: want to add manually
    # Might need to be a copy
    filtered_params = PARAMS.copy()
    filtered_params.pop("vary_param")

    for vary_key, vary_value in filtered_params.items():
        if vary_key in ["vary_param"]:
            continue

        # Set label
        vary["vary_param"].append(vary_key)

        # Set min sample for this param
        vary[vary_key].append(vary_value["min"])
        for const_key, const_value in filtered_params.items():
            # Ignore value we're varying
            if vary_key == const_key:
                continue
            # Add value to be held constant
            vary[const_key].append(const_value["default"])

        vary["vary_param"].append(vary_key)

        # Set max sample for this param
        vary[vary_key].append(vary_value["max"])
        for const_key, const_value in filtered_params.items():
            # Ignore value we're varying
            if vary_key == const_key:
                continue
            # Add value to be held constant
            vary[const_key].append(const_value["default"])

    samples = pd.DataFrame(vary)
    # samples
    dataframe_sampler = uq.sampling.DataFrameSampler(samples)
    return dataframe_sampler


def single_param_values(results: pd.DataFrame) -> pd.DataFrame:
    """Calculate range for the violated constraints residuals according to single param method.

    Parameters
    ----------
    results : pd.DataFrame
        Results of the evaluations

    Returns
    -------
    pd.DataFrame
        Absolute range of RMS violated constraint residuals for each varied parameter
    """
    # Extract the single-param values for each varied parameter
    vary_params = results[("vary_param", 0)].unique().tolist()
    vio_constr_res_diff_dict = {"vary_params": [], "vio_constr_res_diffs": []}

    for vary_param in vary_params:
        # Get violated constraint residuals for this varied parameter's min and max samples
        vio_constr_res_series = results[results[("vary_param", 0)] == vary_param][
            "vio_constr_res", 0
        ]
        vio_constr_res = vio_constr_res_series.to_list()
        # Calc diff and add to new dict
        vio_constr_res_diff = abs(vio_constr_res[1] - vio_constr_res[0])
        vio_constr_res_diff_dict["vary_params"].append(vary_param)
        vio_constr_res_diff_dict["vio_constr_res_diffs"].append(vio_constr_res_diff)

    vio_constr_res_diffs_df = pd.DataFrame(vio_constr_res_diff_dict)
    vio_constr_res_diffs_df = vio_constr_res_diffs_df.sort_values(
        "vio_constr_res_diffs", ascending=False
    ).reset_index(drop=True)

    return vio_constr_res_diffs_df


def plot(
    vio_constr_res_diffs_df: pd.DataFrame,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    """Plot single-param values for uncertainties in RMS violated constraint residuals.

    Parameters
    ----------
    vio_constr_res_diffs_df : pd.DataFrame
        Ranges of violated constraint residuals for each parameter

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
        fig and ax of plot for further customisation
    """
    fig, ax = plt.subplots()
    sns.barplot(
        data=vio_constr_res_diffs_df, x="vio_constr_res_diffs", y="vary_params", ax=ax
    )
    ax.set_title(
        "Single-parameter evalutation of uncertainties for generic DEMO solution point"
    )
    ax.set_ylabel("Uncertain parameters")
    ax.set_xlabel("RMS of violated constraint residuals")
    return fig, ax


def non_zero_residuals(vio_constr_res_diffs_df: pd.DataFrame) -> pd.DataFrame:
    """Return subset of RMS violated constraint residuals that are > 0.

    Parameters
    ----------
    vio_constr_res_diffs_df : pd.DataFrame
        RMS violated constraint residuals

    Returns
    -------
    pd.DataFrame
        RMS violated constraint residuals > 0
    """
    return vio_constr_res_diffs_df[vio_constr_res_diffs_df["vio_constr_res_diffs"] > 0]
