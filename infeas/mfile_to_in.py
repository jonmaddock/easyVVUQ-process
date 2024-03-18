"""Convert an output mfile to an input file.

Take the optimisation parameter solution vector from the output mfile and 
use it as the initial optimisation parameter vector in a new input file.
"""

from process.io.mfile import MFile
from process.io.in_dat import InDat
from pathlib import Path
from typing import Optional, Sequence
import re


# List of optimisation param numbers that are f-values
F_VALUE_OPT_PARAM_NUMBERS = [
    14,
    15,
    21,
    25,
    26,
    27,
    28,
    30,
    32,
    33,
    34,
    35,
    36,
    38,
    39,
    40,
    45,
    46,
    48,
    49,
    50,
    51,
    53,
    54,
    62,
    63,
    64,
    66,
    67,
    68,
    71,
    72,
    79,
    86,
    89,
    92,
    95,
    96,
    97,
    103,
    104,
    105,
    106,
    107,
    110,
    111,
    112,
    113,
    115,
    116,
    117,
    118,
    123,
    137,
    141,
    143,
    144,
    146,
    147,
    149,
    153,
    154,
    157,
    159,
    160,
    161,
    164,
    165,
    166,
    167,
    168,
]

# Optimisation parameters that start with f, but are not f-values
STARTS_WITH_F_BUT_NOT_F_VALUE_NAMES = [
    "fcohbop",
    "fvsbrnni",
    "feffcd",
    "fcutfsu",
    "fimp",
    "fgwped",
]


def convert(
    mfile_name: str,
    original_in_name: str,
    sol_in_name: str,
    no_optimisation: Optional[bool] = True,
    n_equalities: Optional[int] = None,
) -> Path:
    """Convert mfile to input file, preserving optimisation parameter vector.

    Parameters
    ----------
    mfile_name : str
        name of mfile to convert
    original_in_name : str
        the IN.DAT used to create the MFILE.DAT solution
    sol_in_name : str
        name of input file to be created at solution vector
    no_optimisation : Optional[bool], optional
        convert IN.DAT to non-optimising run, by default True
    n_equalities : Optional[int], optional
        how many of the constraints to make equalities. The first n_equalities
        constraints will be left as equalities, the rest converted to
        inequalities, by default None

    Returns
    -------
    Path
        path to solution input file
    """
    # First, extract the solution (optimisation parameters) from the mfile
    # Create Mfile object from mfile
    mfile_path = Path(mfile_name)
    mfile = MFile(mfile_path)

    # Get number of optimisation parameters
    n = int(mfile.data["nvar"].get_scan(-1))

    # Get all n optimisation parameter names and values from "itvarxxx" number
    opt_params = {}
    for i in range(n):
        opt_param_no = f"itvar{i+1:03}"
        param_name = mfile.data[opt_param_no].var_description
        param_value = mfile.data[opt_param_no].get_scan(-1)
        opt_params[param_name] = param_value

    # Now overwrite the original input file with the solution as the initial
    # optimisation parameters
    # Read in original IN.DAT
    in_dat_path = Path(original_in_name)
    in_dat = InDat(in_dat_path)

    # Change to the new optimisation parameter values
    for var_name, value in opt_params.items():
        in_dat.add_parameter(var_name, value)

    # Write out new IN.DAT, with optimisation parameters set to original
    # solution vector
    sol_in_dat_path = Path(sol_in_name)
    in_dat.write_in_dat(sol_in_dat_path)

    # Additional modifications to top of input file
    top_content = []
    if no_optimisation:
        top_content.extend(["* Once through only: no optimisation", "ioptimz  = -2"])

    if n_equalities is not None:
        # Write neqns count
        top_content.extend(
            [
                "* Define number of equality constraints, and",
                "* use inequality constraints: corresponding f-values removed",
                f"neqns = {n_equalities}",
            ]
        )

    new_content = [line + "\n" for line in top_content]

    # Write top and main content, removing f-values if necessary
    with open(sol_in_dat_path, "r+") as f:
        content = f.readlines()
        f.seek(0, 0)

        if n_equalities:
            # Remove all f-values
            main_content = remove_f_values(content)
        else:
            main_content = content

        new_content.extend(main_content)
        f.writelines(new_content)

    return sol_in_dat_path


def remove_f_values(lines_with_f_values: Sequence[str]) -> list[str]:
    """Remove f-value optimisation parameters from input file lines.

    Parameters
    ----------
    lines_with_f_values : Sequence[str]
        lines from original input file

    Returns
    -------
    list[str]
        lines with f-value optimisation parameters removed
    """
    f_value_removal_count = 0
    # Lines that will be included in new IN.DAT
    lines_without_f_values = []
    # Opt params beginning with f that aren't on the ignore list
    suspect_opt_params_beginning_with_f = []

    # Read each line searching for optimisation parameter numbers, e.g. ixc = 23
    # Don't also search for comment (e.g. ixc = 23 * fdene) as it may not be there:
    # Just look for opt param number
    opt_param_re = re.compile(r"ixc\s*=\s*(\d+)")
    # If not already found in the f-value opt param number list, attempt to
    # find any opt params starting with f (depends on "*" comment)
    opt_param_beginning_with_f_re = re.compile(r"ixc\s*=\s*(\d+)\s*\*\s*(f\w+)")

    for line in lines_with_f_values:
        matches = opt_param_re.match(line)
        if matches is not None:
            # Found an opt param line
            opt_param_number = matches.group(1)
            if int(opt_param_number) in F_VALUE_OPT_PARAM_NUMBERS:
                # Found an f-value: drop the line
                line_fmt = line.removesuffix("\n")
                print(f'Removing "{line_fmt}"')
                f_value_removal_count += 1
                continue

        # Non f-value opt param or other line: keep
        lines_without_f_values.append(line)

        # Check if line contains opt param beginning with an f:
        # Not known f-value (would be in numbers list), but suspicious as could be
        matches = opt_param_beginning_with_f_re.match(line)
        if matches is not None:
            # Check name not on known "not an f-value" list
            if matches.group(2) not in STARTS_WITH_F_BUT_NOT_F_VALUE_NAMES:
                # Opt param, starts with f, not known as a "non-f-value" var
                # Could be an f-value
                suspect_opt_params_beginning_with_f.append(line)

    # Print out potential f-value vars if any
    if len(suspect_opt_params_beginning_with_f) > 0:
        raise ValueError(
            f"Suspect opt params to check aren't f-values: {suspect_opt_params_beginning_with_f}"
        )
    else:
        print(f"{f_value_removal_count} f-values removed")

    # TODO Remove f-value values too: all should be >= 1

    return lines_without_f_values
