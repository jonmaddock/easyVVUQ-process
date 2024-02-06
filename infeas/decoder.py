"""Custom decoder for Process's mfiles.

Extracts data from Process's mfiles for easyVVUQ to use.
"""

import easyvvuq as uq
import numpy as np
from typing import Dict, Any, Optional
from process.io.mfile import MFile
import regex as re
from pathlib import Path


class MfileDecoder(uq.decoders.JSONDecoder):
    """Interprets Process's mfiles to extract repsonses for easyVVUQ.

    Subclasses easyVVUQ's JSON decoder.
    """

    def _get_raw_data(self, out_path: str) -> Dict:
        """Parse mfile and return dictionary of all output data.

        Parameters
        ----------
        out_path : str
            Path to mfile

        Returns
        -------
        Dict
            All output data contained in mfile
        """
        mfile = MFile(Path(out_path))
        mfile_dict = {}
        for param_name in mfile.data:
            param_value = mfile.data[param_name].get_scan(-1)
            mfile_dict[param_name] = param_value

        return mfile_dict

    def _process_raw_data(self, raw_data: Dict[str, Any]) -> Dict[str, float]:
        """Perform any required processing of raw mfile data dict.

        May include filtering for desired responses.

        Parameters
        ----------
        raw_data : Dict[str, Any]
            Entire raw output dictionary

        Returns
        -------
        Dict[str, float]
            Processed output dictionary
        """
        # Add objective function to responses dict
        responses = {"norm_objf": raw_data["norm_objf"]}

        # Extract constraints from raw_data dict
        # Find all equality and inequality constraint keys
        eq_re = re.compile(r"eq_con\d{3}")
        ineq_re = re.compile(r"ineq_con\d{3}")
        eq_constrs_keys = [key for key in raw_data.keys() if eq_re.match(key)]
        ineq_constrs_keys = [key for key in raw_data.keys() if ineq_re.match(key)]

        # Get values of constraints
        eq_constrs_dict = {
            eq_constrs_key: raw_data[eq_constrs_key]
            for eq_constrs_key in eq_constrs_keys
        }
        ineq_constrs_dict = {
            ineq_constrs_key: raw_data[ineq_constrs_key]
            for ineq_constrs_key in ineq_constrs_keys
        }

        # Only want violated constraint values
        # Coerce feasible inequality constraints (> 0) = 0.0
        # TODO Not sure if we want to mask non-violated constraint
        # values at this stage: infeasibile responses only
        vio_ineq_constrs_dict = {}
        for key, value in ineq_constrs_dict.items():
            if value > 0:
                vio_ineq_constrs_dict[key] = 0.0
            else:
                vio_ineq_constrs_dict[key] = value

        # Merge individual eq and ineq violated constraint values
        responses = responses | eq_constrs_dict | vio_ineq_constrs_dict

        # Calculate RMS constraint residuals for violated constraints only
        # Create arrays from constraints dicts
        eq_constrs = np.array(list(eq_constrs_dict.values()))
        vio_ineq_constrs = np.array(list(vio_ineq_constrs_dict.values()))
        vio_constrs = np.concatenate((eq_constrs, vio_ineq_constrs))
        rms_vio_constr_res = np.sqrt(np.mean(vio_constrs**2))
        responses["rms_vio_constr_res"] = rms_vio_constr_res

        return responses

    def parse_sim_output(self, run_info: Optional[Dict] = None) -> Dict[str, float]:
        """Parse mfile, process it and return dict to easyVVUQ.

        Adapted from JSON decoder source to include _process_raw_data() step.

        Parameters
        ----------
        run_info : Optional[Dict], optional
            Run information supplied by easyVVUQ, by default None

        Returns
        -------
        Dict[str, float]
            Response data for easyVVUQ

        Raises
        ------
        RuntimeError
            Raised if field is absent from processed output data
        """
        if run_info is None:
            run_info = {}

        def get_value(data, path):
            for node in path:
                data = data[node]
            return data

        out_path = self._get_output_path(run_info, self.target_filename)
        raw_data = self._get_raw_data(out_path)

        # Perform any required processing of raw data
        processed_data = self._process_raw_data(raw_data)

        data = []
        for col in self.output_columns:
            try:
                if isinstance(col, str):
                    data.append((col, processed_data[col]))
                elif isinstance(col, list):
                    data.append((".".join(col), get_value(processed_data, col)))
            except KeyError:
                raise RuntimeError("no such field: {} in this mfile".format(col))
        return dict(data)
