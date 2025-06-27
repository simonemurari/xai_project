## XAI Project C51 + Rules on OfficeWorld

In this repository you can find the code to run the C51 and DQN algorithms augmented with rules on the OfficeWorld environment. The code of the algorithms is based on the original implementation of CleanRL. In `env_README.md` you can find the original README.md of the OfficeWorld environment and in `gym-subgoal-automata` you can find the code for the environments.

### Repository Structure

- **c51_rules.py**:  
  Contains the implementation of the C51 algorithm with additional rules applied during training.

- **dqn_rules.py**:  
  Contains the implementation of the DQN algorithm with additional rules applied during training.

### How to run

1. **Create a Python environment**  
   Create a virtual environment with Python 3.7.9 and install the packages. For example, using conda:
   ```bash
   conda create -n xai-project-officeworld python=3.7.9
   conda activate xai-project-officeworld
   pip install -r requirements.txt
   ```
   Or you can install `uv` and do `uv sync`.

2. **Configure wandb (optional)** \
   If you want to use wandb tracking, create a `.env` file in the repository root and set the proper environment variables (WANDB_KEY, WANDB_PROJECT_NAME, WANDB_ENTITY).

3. **Configure the config.py file** \
    For `c51_rules.py` you need to set the respective parameters in the `config.py` file while for `dqn_rules.py` you need to set the parameters in `config_dqn.py`.

4. **Run the scripts** \
   To run the different scripts just do e.g.: ```python c51_rules.py``` or ```uv run c51_rules.py``` if you are using `uv` 
