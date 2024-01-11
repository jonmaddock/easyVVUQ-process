# easyVVUQ-PROCESS: Feasibility Uncertainty Quantification (UQ)

A feasibility study using the PROCESS nuclear fusion power plant systems code and the easyVVUQ framework. It aims to assess the feasibility of two different power plant concepts under epistemic uncertainty.

A DEMO-like design is optimised using PROCESS for maximum net electric output, and a second design optimised for minimum major radius. UQ studies are then performed on these two design points to assess their comparative feasibilities under uncertainty.

## Installation

It is recommended to use a virtual environment. Firstly, Process needs to be installed. Clone the `process-uq` fork of Process, and switch to the `dakota` branch. This branch adds the `responses.json` output file, which contains the violated constraint residuals.
```
git clone -b dakota https://github.com/jonmaddock/process-uq.git
```

Then install Process:
```
cd process-uq
cmake -S . -B build
cmake --build build
```

Secondly, clone this repository and install the `infeas` package:
```
cd ..
git clone https://github.com/jonmaddock/process-uq.git
cd process-uq
pip install .
```

Whether installed locally or on HPC, this should allow any of the notebooks in this repository to run.