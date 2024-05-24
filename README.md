<h2 style="text-align: center;">Understanding the differences in Foundation Models:<br> Attention, State Space Models, and Recurrent Neural Networks</h2>

This repository contains the code of the paper titled "Understanding the differences in Foundation Models: Attention, State Space Models, and Recurrent Neural Networks".

This code is a derivative of the [Zoology code base](https://github.com/HazyResearch/zoology).

### Getting started
1. Install Zoology using the install instructions in the original README copied below.
1. Install the selective scan CUDA implementation from `mamba-ssm`, see [here](https://github.com/state-spaces/mamba). This is needed for running S6 and SSD efficiently.
1. Install `accelerated-scan` from [here](https://github.com/proger/accelerated-scan). This is needed for running the qLSTMs efficiently.

### Reproducing paper experiments
The configs, instructions and plotting code for reproducing the figures in these papers are provided in the following sub-folders. 

- configs
    - `zoology/experiments/dsf-arxiv/`
- plotting
    - `../notebooks/plotting.ipynb`

-----

<div align="center" >
    <img src="assets/banner.png" height=150 alt="Meerkat logo" style="margin-bottom:px"/> 

**Understand and test language model architectures on synthetic tasks.**


</div>


## Getting started

**Installation.** First, ensure you have torch installed, or install it following the instructions [here](https://pytorch.org/get-started/locally/). Then, install Zoology with:
 
```bash
git clone https://github.com/HazyResearch/zoology.git
cd zoology
pip install -e .[extra,analysis] 
```
If you want to keep this install as lightweight as possible; the only required dependencies are: `torch, einops, tqdm, pydantic, wandb`. There is some extra functionality (*e.g.* launching sweeps in parallel with Ray) that require additional dependencies. To install without the optional dependencies, run `pip install -e .`.

Then, try running an example experiment with: 
```
python -m zoology.launch zoology/experiments/examples/basic.py
```
This will train a simple two layer transformer on multi-query associative recall. To run a sweep over learning rates, try: 
```
python -m zoology.launch zoology/experiments/examples/basic_sweep.py
```
If you have access to multiple GPUs, you can run the sweep in parallel by adding the `-p` flag.


## Configuration, Experiments, and Sweeps
In this section, we'll walk through how to configure an experiment and launch sweeps. 

*Configuration*. Models, data, and training are controlled by configuration objects. For details on available configuration fields, see the configuration definition in [`zoology/config.py`](zoology/config.py). The configuration is a nested Pydantic model, which can be instantiated as follows:
```python
from zoology.config import TrainConfig, ModelConfig, DataConfig, ModuleConfig, FunctionConfig

config = TrainConfig(
    max_epochs=20,
    data=DataConfig(
        train_configs=[MQARConfig(num_examples=10_000, vocab_size=128, input_seq_len=input_seq_len, **factory_kwargs)],
        test_configs=[MQARConfig(num_examples=1_000, vocab_size=128, input_seq_len=input_seq_len, **factory_kwargs)],
    ),
    model=ModelConfig(
        vocab_size=128,
        sequence_mixer=ModuleConfig("name": "zoology.mixers.attention.MHA"}
    ),
)
```
Note that the `FunctionConfig` and `ModuleConfig` are special objects that configure partial functions and PyTorch modules, respectively. 
They both have an `instantiate()` method that will import the function or class passed to `name` and partial or instantiate it with `kwargs`.
For example, 
```python
fn_config = FunctionConfig(name="torch.sort", kwargs={"descending": True})
fn = fn_config.instantiate()
fn(torch.tensor([2,4,3])) # [4, 3, 2]
```

*Launching experiments.* To launch an experiment from the command line, define a configuration object in python file and store it in a global variable `configs`:
```python
config = TrainConfig(...)
configs = [config]
```
See [`zoology/experiments/examples/basic.py`](zoology/experiments/examples/basic.py) for an example. 

Then run `python -m zoology.launch zoology/experiments/examples/basic.py`, replacing `basic.py` with the path to your experiment. This will launch a single training job. 


*Launching sweeps.* To launch a sweep, simply add more configuration objects to the `configs` list. For example, here's the content of [`zoology/experiments/examples/basic_sweep.py`](zoology/experiments/examples/basic_sweep.py):
```python
import numpy as np
from zoology.config import TrainConfig

configs = []
for lr in np.logspace(-4, -2, 10):
   configs.append(TrainConfig(learning_rate=lr)) 
```
You can then run `python -m zoology.launch zoology/experiments/examples/basic_sweep.py`. This will launch a sweep with 10 jobs, one for each configuration.

*Launching sweeps in parallel.* If you have multiple GPUs on your machine, you can launch sweeps in parallel across your devices. 
To launch sweeps in parallel, you'll need to install [Ray](https://docs.ray.io/en/latest/ray-overview/installation.html): `pip install -e.[extras]`. 
Then, you can run `python -m zoology.launch zoology/experiments/basic_sweep.py -p`. 
This will run the configurations in parallel using a pool of workers, one per GPU.

*Logging.* Zoology uses [Weights and Biases](https://wandb.ai/site) for logging. You'll need to login with `wandb login` and update the `LoggerConfig` in your configuration to point to your project: 
```python
from zoology.config import TrainConfig, LoggerConfig

TrainConfig(
    logger=LoggerConfig(
        project="my_wandb_project",
        entity="my_wandb_entity",
    ),
    ...
)
```

## Data
In this section, we'll walk through how to create a new synthetic task and discuss some of the tasks that are already implemented.

*Creating a new task.* To create a new task, you'll need to subclass `zoology.config.DataSegmentConfig`. 
See zoology/data/associative_recall.py  for an example. 
```python
class DataSegmentConfig(BaseConfig):
    """
    This class should be subclassed to define per task. For example, MQARConfig
    """
    vocab_size: int = 8_192
    num_examples: int = 1_000
    input_seq_len: int = 64

    def build(self, **kwargs):
        raise NotImplementedError()
```

You'll need to implement the `build` method, which should return a `zoology.data.utils.DataSegment` object, a simple dataclass:

```python
@dataclass
class DataSegment:
    inputs: torch.Tensor
    labels: torch.Tensor
    slices: Dict[str, any] = None
```
The inputs and labels should be integer tensors with values in the range `[0, vocab_size)`. 


You can create this subclass in any file you want, as long as it's importable. Let's
assume that we've created a file `zoology/data/my_task.py` and written our `MyDataSegmentConfig` function there.
Then, we can add it to our data configuration with: 
```python
from zoology.config import TrainConfig, DataConfig, FunctionConfig
config = TrainConfig(
    DataConfig(
        train_configs=[MyDataSegmentConfig(num_examples=10_000, vocab_size=128, input_seq_len=input_seq_len, **other_kwargs)],
        test_configs=[MyDataSegmentConfig(num_examples=1_000, vocab_size=128, input_seq_len=input_seq_len, **other_kwargs)],
    ),
)
```


**Caching dataset creation.** Sometimes it's useful to cache the dataset creation process, especially if it's expensive. To do so you can pass a `cache_dir` to the `DataConfig`: `DataConfig(..., cache_dir="my_cache_dir")`.

