# Infeasibility UQ examples

## Local evaluation

If all uncertainties (21) are allowed to vary simultaneously using 3rd-order PCE sampling, this results in 1.04e6 samples, rendering the problem very intensive. This therefore motivates a parameter screening study to identify the most influential parameters on the response, so that a subsequent detailed study can use these influential "screened" parameters only to make the problem computationally tractable. By using only the 4 most influential uncertain parameters, the study can be run locally.

This study runs a proof-of-concept sensitivity analysis on Process. Infeasibility UQ analysis can be run locally, e.g. on a laptop, by using few uncertain parameters. The evaluation notebook `infeas_screened_eval_local.ipynb` performs the sample evaluations (intensive), whilst the analysis notebook `infeas_screened_analysis_local.ipynb` plots the results from the generated database (non-intensive). Reponses include the individual violated constraint residuals as well as the RMS violated constraints.

This study uses a DEMO-like design point (optimised for maximum net electric power) with 4 epistemic uncertainties. Process's models are run once-through for a given value of uncertain inputs according to the Polynomial Chaos Expansion (PCE) sampling method, and the response/quantity of interest is the value of constraints, i.e. the feasibility. The results are the sensitivities (Sobol indicies), presented as a treemap plot.

f-values (extra optimisation parameters used to convert the inequality constraints to equalities to support the legacy solver HYBRD) in the input file are fixed to 1.0; this causes the equality constraints to become inequalities again. This means that the values of the inequalities can be used to assess how satisfied/violated the constraints are.

## Cluster evaluation

For more intensive studies (more uncertain parameters, higher order PCEs), parallelised cluster execution is required. The job script `run_infeas_cluster.job` only requests a single processor, but Dask is subsequently used to parallelise to 4 "workers" (in this case, processors) in the `infeas_cluster.ipynb` notebook. This approach is used in the main study.