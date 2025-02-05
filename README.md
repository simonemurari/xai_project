## XAI Project C51 + Rules on Minigrid
In this repository you can find the code to run the C51 and C51 + Rules algorithm on the Minigrid environment from Gymnasium. The code is from CleanRL and was modified to work with the Minigrid environment. \
The main files are `c51.py`, `c51_eval.py`, `c51_rules_training.py` and `c51_rules_eval.py`.
The first two files are used to train and evaluate the C51 algorithm while the last two files apply the rules to the C51 algorithm: in `c51_rules_training.py` the rules are applied only during training while in `c51_rules_eval.py` the rules are applied only during evaluation on networks trained with the base C51 algorithm. \
The parameters for the algorithms can be set in the `config.py` file.

### Installation
To run the code you need to create a virtual environment and install the required packages, you can do this with `conda` by running the following commands:
```bash
conda create -n xai-project python=3.10.12
conda activate xai-project
pip install -r requirements.txt
python c51.py
```
Or you can install `uv` and do:
```bash
uv run c51.py
```

### Results
You can find the results in the `C51/runs` or `C51rt/runs_rules_training` folders based on the algorithm you used. To visualize the results you can run tensorboard with the following command:
```bash
tensorboard --logdir runs
```
Or you can just look at the images in the folders `C51`, `C51rt`, `C51re`.
It was tested on Python 3.10.12 on WSL2 on Windows 11 and Python 3.10.12 on Windows 11.