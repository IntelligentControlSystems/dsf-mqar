
## Reproducing paper experiments
To reproduce the paper results, ensure you have WandB setup to log all the results and then run the command:
```
python -m zoology.launch zoology/experiments/neurips24/<config>.py
```
where `<config>` is any of the 7 config files in the directory.
Note that there are 1304 model/data configurations in this sweep, so it takes a while to run. We ran most of our experiments on a cluster with the `-p` flag, which launches configurations in parallel. To run a smaller scale experiment, you can modify the loops in `<config>.py` file to only include a subset of the configurations you're interested in (*e.g.* you can drop some models, sequence lengths, or learning rates). For more details on how the experiments are configured, see the details in the paper.

To produce the plots after the run, see the plotting notebook `notebooks/plotting.ipynb`.