## XAI Project C51
In this repository you can find the code to run the C51 algorithm on the Minigrid environment from Gymnasium. The code is from CleanRL and was modified to work with the Minigrid environment. \
The parameters for the C51 algorithm can be found in the `config.py` file while the main code is in the `c51.py` file. \
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
You can find the results in the `runs` folder and view them using tensorboard:
```bash
tensorboard --logdir runs
```
Or you can just look at the images in the `images` folder to see the episodic return over time. \
It was tested on Python 3.10.12 on WSL2 on Windows 11 and Python 3.10.12 on Windows 11.