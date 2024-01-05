# Maximum net electric power
Sensitivity analysis and uncertainty quantification on the maximum net electric power DEMO-like solution under epistemic uncertainty.

First, perform a parameter screening analysis to find the most influential uncertain parameters on the response (constraint violations). Then perform a Polynomial Chaos Expansion (PCE) on these most influential parameters, now the problem is computationally tractable. The result is the sensitivities (Sobol indicies) and uncertainties on the constraint violations for the uncertaint input parameters.

- The `local` directory runs inexpensive screening, PCE evaluation and analysis tractable on a local machine
- `screening` contains the full parameter screening analysis, to identify the most influential parameters
- `screened` contains the PCE analysis on the screen parameters, requiring HPC
- `*.template` is the template input file, which is the DEMO-like solution point with placeholders for the uncertain parameters
- `dask_uq_template.py` is a template for running easyVVUQ using Dask for parallelism on a SLURM cluster

