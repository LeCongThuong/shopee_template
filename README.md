# Deep Learning project template
Use this template to rapidly bootstrap a DL project:

- Write code in [Pytorch Lightning](https://www.pytorchlightning.ai/)'s `LightningModule` and `LightningDataModule`.
- Run code from composable `yaml` configurations with [Hydra](https://hydra.cc/).
- Manage packages in `environment.yaml` with [conda](https://docs.conda.io/projects/conda/en/latest/glossary.html#miniconda-glossary).
- Log and visualize metrics + hyperparameters with [Tensorboard](https://tensorboard.dev/).
- Sane default with best/good practices only where it makes sense for small-scale research-style project.

Have an issue, found a bug, know a better practice? Feel free to open an issue, pull request or discussion thread. All contribution welcome.

I hope to maintaining this repo with better deep learning engineering practices as they evolve.

## Quick start

<details><summary>Click to expand/collapse</summary>
<p>
    
### 0. Clone this template
```bash
# clone project or create a new one from GitHub's template
git clone https://github.com/lkhphuc/lightning-hydra-template new-project
cd new-project
rm -rf .git
git init  # Start of a new git history
```

### 1. Add project's info
- Edit [`setup.py`](setup.py) and add relevant information.
- Rename the directory `project/` to the your project name.

### 2. Create environment and install dependencies
- Name your environment and add packages in [`environment.yaml`](environment.yaml), then create/update the environment with:
```bash
# Run this command every time you update environment.yaml
conda env update -f environment.yaml
```

### 3. Create Pytorch Lightning modules
- `LightningModule`s are organized under [`project/model/`](project/model/).
- `LightningDataModule`s are organized under [`project/data/`](project/data/).

Each Lightning module should be in one separate file, while each file can contain all the relevant `nn.Module`s for that model.

### 4. Create Hydra configs
Each `.py` file has its own corresponding `.yaml` file, such as `project/model/autoencoder.py` and `configs/model/autoencoder.yaml`.

All `yaml` files are stored under `configs/` and the structure of this folder should be identical to the structure of the `project/`.
```bash
$ tree project              $ tree configs
project                     configs
├── __init__.py             ├── defaults.yaml
├── data                    ├── data
│   ├── cifar.py            │   ├── cifar.yaml                              
│   └── mnist.py            │   └── mnist.yaml
└── model                   ├── model
    ├── autoencoder.py      │   ├── autoencoder.yaml
    ├── classifier.py       │   └── classifier.yaml
                            └── optim
                                ├── adam.yaml
                                └── sgd.yaml
```
[`configs/defaults.yaml`](configs/defaults.yaml) contains all the defaults modules and arguments, including that for the `Trainer()`.


### 5. Run
```bash
# This will run with all the default arguments
python main.py
# Override defaults from command line
python main.py model=autoencoder data=cifar trainer.gpus=8
```
</p>
</details>

## How it works
This section will provide a brief introduction on how these components all come together. 
Please refer to the original documents of [Pytorch Lightning](pytorchlightning.ai/), [Hydra](hydra.cc/) and [TensorBoard](tensorboard.dev) for details.

<details><summary>Click to expand/collapse</summary>
<p>
    
### Entry points
The launching point of the project is [`main.py`](main_2.py) located in the root directory.
The `main()` function takes in a `DictConfig` object, which is prepared by `hydra` based on the `yaml` files and command line arguments provided at runtime.

This is achieved by decorating the script `main()` function with `hydra.main()`, which requires a path to all the configs and a default `.yaml` file as follow:
```python
@hydra.main(config_path="configs", config_name="defaults")
def main(cfg: DictConfig) -> None: ...
```
This allow us to define multiple entry points for different functionalities with different defaults, such as `train.py`, `ensemble.py`, `test.py`, etc.


### Dynamically instantiate modules
We will [use Hydra to instantiate objects](https://hydra.cc/docs/patterns/instantiate_objects/overview).
This allow us to use the same entry point (`main.py` above) to dynamically combine different models and data modules.
Given a [`configs/defaults.yaml`](configs/defaults.yaml) file contains:
```yaml
defaults:
  - data: mnist  # Path to sub-config, can also omit the .yaml extension
  - model: classifier.yaml  # full path for ease of navigation (e.g vim cursor in path, press gf)
```

Different modules can be instantiated for each run by supplying a different set of configuration:
```bash
# Using default
$ python main.py 

# The default is equivalent to
$ python main.py model=classifier data=mnist

# Override a default module
$ python main.py model=autoencoder
$ python main.py data=cifar

# Override multiple default modules and arguments
$ python main.py model=autoencoder data=cifar trainer.gpus=4
```

In python, the module will be instantiated by a line, for example `data_module = hydra.utils.instantiate(cfg.data)`.

`cfg.data` is a `DictConfig` object created by `hydra` at runtime, and is stored in a config file, for example [`configs/data/mnist.yaml`](configs/dataset/mnist.yaml):
```yaml
name: mnist

# _target_ class to instantiate
_target_: project.dataset.MNISTDataModule
# Argument to feed into __init__() of target module
data_dir: ~/datasets/MNIST/  # Use absolute path
batch_size: 4
num_workers: 2

# Can also define arbitrary info specific to this module
input_dim: 784
output_dim: 10
```
and the _target_: `project.data.MNISTDataModule` to be instantiated is:
```python
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "",
                       batch_size: int = 32,
                       num_workers: int = 8,
                **kwargs): ...
# kwargs is used to handle arguments in the DictConfig but not used for init
```

### Directory management
Since `hydra` manages our entry point and command line arguments, it also manages the output directory of each run.
We can easily customize the output directory to suit our project via [`defaults.yaml`](configs/defaults.yaml)
```yaml
hydra: 
  run:
    # Configure output dir of each experiment programmatically from the arguments
    # Example "outputs/mnist/classifier/baseline/2021-03-10-141516"
    dir: outputs/${dataset.name}/${model.name}/${experiment}/${now:%Y-%m-%d_%H%M%S}
```
and tell `TensorBoardLogger()` to use the current working directory without adding anything:
```python
tensorboard = pl.loggers.TensorBoardLogger(".", "", "")
```

</p>
</details>

## Best practices

<details><summary>Click to expand/collapse</summary>
<p>
    
### `LightningModule` and `LightningDataModule`
#### Be explicit about input arguments
Each modules should be self-contained and self-explanatory, to maximize reusability, even across projects.
- **Don't** do this:
```python
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
```
You will not like it when having to track down the config file every time just to remember what are the input arguments, their types and default values.

- Do this instead:
```python
class LitAutoEncoder(pl.LightningModule):
    def __init__(self,
        input_dim: int, output_dim: int, hidden_dim: int = 64,
        optim_encoder=None, optim_decoder=None,
    **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # Later all input arguments can be accessed anywhere by
        self.hparams.input_dim
        # Use this to avoid boilderplate code such as
        self.input_dim = input_dim
        self.output_dim = output_dim
```


Also see Pytorch Lightning's [official style guide](https://pytorch-lightning.readthedocs.io/en/latest/starter/style_guide.html).

### Tensorboard
- Use forward slash `/` in naming metrics to group it together.
    - Don't: `loss_val`, `loss_train`
    - Do:    `loss/val`, `loss_train`
- Group metrics by type, not on what data it was evaluate with:
    - Don't: `val/loss`, `val/accuracy`, `train/loss`, `train/acc`
    - Do:   `loss/val`, `loss/train`, `accuracy/val`, `accuracy/train`
    ![Metric grouping](https://pytorch.org/docs/stable/_images/hier_tags.png)
- Log computation graph of `LightningModule` by:
    - Define `self.example_input_array` in your module's `__init__()`
    - Enable in TensorBoard with `TensorBoard(log_graph=True)`
    ![Compute Graph](https://raw.githubusercontent.com/tensorflow/tensorboard/master/docs/images/graphs_conceptual.png)
- [Proper loggin](https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#logging-hyperparameters) of hyper-parameters and metrics
    ![Tensorboard Parallel Coordinate](https://www.tensorflow.org/tensorboard/images/hparams_parallel_coordinates.png)


### Hydra

#### Script is for one run, launcher is for multiple run
Hydra serves two intertwined purposes, configuration management and script launcher.
These two purposes are dealt with jointly because each run can potentially has a different set of configs.

This provides a nice separation of concerns, in which the python scripts only focus on the functionalities of individual run, while the `hydra` command line will orchestrate multiple runs.
With this separation, it's easy to use Hydra's [sweeper](https://hydra.cc/docs/plugins/ax_sweeper) to do hyperparameters search, or [launcher](https://hydra.cc/docs/plugins/submitit_launcher) to run experiments on SLURM cluster or cloud.

#### Provide absolute path in config
To provide path into program, it's best to provide an absolute path for both local or cloud storage (start with `~`, `/`, `s3://`).

That way you don't have litter your code with `hydra.utils.get_original_cwd()` to convert relative path, and therefore retaining the flexibility to use your module outside of `hydra`-managed entry points.

#### Naming experiments
Use `hydra` to created a hierarchical structure for experiments output based on configurations of each run, by setting the `configs/defaults.yaml` with 
```
dir: outputs/${data.name}/${model.name}/${experiment}/${now:%Y-%m-%d_%H%M%S}
```

- `${data.name}/${model.name}` will be dynamically determined from config object. They are preferably nested by the order of least frequently changed.
- `${experiment}` is a string briefly describe the purpose of the experiment
- `${now:%Y-%m-%d_%H%M%S}` will insert the time of run, serves as a unique identifier for runs differ only in minor hyperparameters such as learning rate.

Example output:`outputs/mnist/classifier/baseline/2021-03-10-141516`.


</p>
</details>

## Tips and tricks

<details><summary>Click to expand/collapse</summary>
<p>
    
### Debug
    
- Drop into a debugger anywhere in your code with a single line `import pdb; pdb.set_trace()`.
- Use `ipdb` or [pudb](github.com/inducer/pudb) for nicer debugging experience, for example `import pudb; pudb.set_trace()`
- Or just use `breakpoint()` for Python 3.7 or above. Set `PYTHONBREAKPOINT` environment variable to make `breakpoint()` use `ipdb` or `pudb`, for example `PYTHONBREAKPOINT=pudb.set_trace`.
- Post mortem debugging by running script with `ipython --pdb`. It opens a debugger and drop you right into when and where an Exception is raised.
```bash
$ ipython --pdb main.py -- model=autoencoder
```
This is super helpful to inspect the variables values when it fails, without having to put a breakpoint and then run the script again, which can takes a long time to start for deep learning model.
- Use `fast_dev_run` of PytorchLightning, and checkout the entire [debugging tutorial](https://pytorch-lightning.readthedocs.io/en/stable/common/debugging.html).

### Colored Logs
    
It's 2021 already, don't squint at your 4K HDR Quantum dot monitor to find a line from the black & white log.
`pip install hydra-colorlog` and edit `defaults.yaml` to colorize your log file:
```yaml
defaults:
  - override hydra/job_logging: colorlog
  - override hydra/hydra_logging: colorlog
```
This will colorize any python logger you created anywhere with:
```python
import logging
logger = logging.getLogger(__name__)
logger.info("My log")
```

Alternative: [loguru](https://github.com/Delgan/loguru), [coloredlogs](https://github.com/xolox/python-coloredlogs).

### Auto activate conda environment and export variables
    
[Zsh-autoenv](https://github.com/Tarrasch/zsh-autoenv) will auto source the content of `.autoenv.zsh` when you `cd` into a folder contains that file.
Say goodbye to activate conda or export a bunch of variables for every new terminal:
```bash
conda activate project
HYDRA_FULL_ERROR=1
PYTHON_BREAKPOINT=pudb.set_trace
```

Alternative: https://github.com/direnv/direnv, https://github.com/cxreg/smartcd, https://github.com/kennethreitz/autoenv

</p>
</details>


## TODO
- [ ] Pre-commit hook for python `black`, `isort`.
- [ ] [Experiments](https://hydra.cc/docs/next/patterns/configuring_experiments) 
- [ ] Configure trainer's callbacks from configs as well.
- [ ] [Structured Configs](https://hydra.cc/docs/next/tutorials/structured_config/intro/#internaldocs-banner)
- [ ] [Hydra Torch](https://github.com/pytorch/hydra-torch) and [Hydra Lightning](https://github.com/romesco/hydra-lightning)
- [ ] [Keepsake](https://keepsake.ai/) version control
- [ ] (Maybe) Unit test (only where it makes sense).


# DELETE EVERYTHING ABOVE FOR YOUR PROJECT  
 
---

<div align="center">
 
# ConSelfSTransDRLIB:
## Contrastive Self-supervised Transformers for Disentangled Representation Learning with Inductive Biases is All you need, and where to find them.

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)

</div>
 
## Description   
Code for paper paper.

## How to run 
```bash
python main.py
```


### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
