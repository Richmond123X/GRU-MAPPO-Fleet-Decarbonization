# GRU-MAPPO Fleet Decarbonization

Research code for multi-agent reinforcement learning in fleet decarbonization under IMO mid-term measures. The repository contains the simulation environment, GRU-MAPPO, MAPPO and QMIX implementations, and the training entry point.

## Contents

- `config.py`: fuel, policy, vessel and simulation parameters.
- `ship.py`: ship and fleet state models.
- `environment.py`: fleet decision-making environment.
- `mappo.py`: MAPPO networks and agent.
- `gru_mappo.py`: recurrent critic and GRU-MAPPO agent.
- `qmix.py`: QMIX baseline.
- `integrated_training.py`: training, checkpoint handling and evaluation utilities.
- `load_real_ships.py`: vessel data loaders.
- `reward_optimizer.py`: reward processing utilities.

## Installation

Use Python 3.11 in a virtual environment:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the virtual environment before installing dependencies. For GPU execution, install the PyTorch build appropriate for your CUDA environment.

## Input data

Datasets and model checkpoints are not distributed in this repository. Supply your own vessel data before training. An Excel workbook accepted by `load_ships_from_excel` must contain these columns:

`Size`, `Fuel_Type`, `Build_Year`, `ME`, `Name`, `IMO`, `MMSI`, `DWT`.

`Size` is vessel capacity in TEU and `ME` is main-engine power in kW. The CSV loader uses a separate schema defined in `load_real_ships.py`; renaming an Excel file to CSV is not sufficient.

## Training

Run commands from the repository directory. For example, with a locally supplied workbook:

```bash
python integrated_training.py --algorithm gru-mappo --mode train --ships_csv_path "path/to/Container_ship.xlsx" --num_ships 100 --iterations 1000 --seed 42 --device cpu --save_dir "outputs/gru_mappo"
```

This is an example invocation; set the fleet size, iteration count and other options for your intended run. Use `--algorithm mappo` or `--algorithm qmix` for the included baselines. Other algorithm names displayed by the existing command-line parser require optional modules not included in this minimal code release.

Inspect available arguments with:

```bash
python integrated_training.py --help
```

Training outputs and checkpoints are written locally. Input data, checkpoints, generated results and manuscript files should not be committed to this repository.
