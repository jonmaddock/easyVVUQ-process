"""Convert an output mfile to an input file.

Take the optimisation parameter solution vector from the output mfile and 
use it as the initial optimisation parameter vector in a new input file.
"""

from process.io.mfile import MFile
from process.io.in_dat import InDat
from pathlib import Path


def convert(mfile_name: str, original_in_name: str, sol_in_name: str) -> Path:
    """Convert mfile to input file, preserving optimisation parameter vector.

    Parameters
    ----------
    mfile_name : str
        name of mfile to convert
    original_in_name : str
        the IN.DAT used to create the MFILE.DAT solution
    sol_in_name : str
        name of input file to be created at solution vector

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

    return sol_in_dat_path
