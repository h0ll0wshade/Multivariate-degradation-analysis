# NASA C-MAPSS data

This directory contains the C-MAPSS turbofan engine degradation benchmark files. Each row records an engine ID, an operating cycle, three operating settings, and 21 sensor readings.

The training trajectories run through failure. Test trajectories stop before failure, and each `RUL_FD00*.txt` file gives the remaining cycles for the corresponding test engines. The four subsets differ in operating conditions and fault modes; see the accompanying `readme.txt` for the supplied description.

The current univariate analysis uses `train_FD001.txt` and one raw signal (`sensor_11`). It does not combine sensors with PCA. Dataset background: Saxena et al., “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,” PHM 2008.
