## XAI Project C51 + Rules on MiniGrid

In this repository you can find the code to run the C51 algorithm and its variations using additional rules on the MiniGrid environment from Gymnasium. The code is adapted from CleanRL to work with MiniGrid.

### Repository Structure

- **baseC51**:  
  Contains the implementation of the normal C51 algorithm without any additional rules.

- **rules_in_training**:  
  Contains the implementations where the rules are applied during training. The models in these folders include variations that modify the training process using predefined rules.

- ~~**rules_in_eval**~~:  
  ~~Contains the implementations where the rules are applied during evaluation. In these scripts the rules are applied on models trained with the base C51 algorithm.~~

- **config.py**:  
  This file is used to set the different parameters for each run (learning rate, batch size, environment settings, etc.).

### How to run

0. **Clone this repository**

1. **Create a Python environment**  
   Create a virtual environment with Python 3.12.6 and install the packages. For example, using conda:
   ```bash
   conda create -n xai-project python=3.12.6
   conda activate xai-project
   pip install -r requirements.txt
   ```

   Or you can install `uv` and do `uv sync`.
2. **Configure wandb (optional)** \
   If you want to use wandb tracking, create a `.env` file in the repository root and set the proper environment variables (WANDB_KEY, WANDB_PROJECT_NAME, WANDB_ENTITY).

3. **Configure the config.py file** \
   In the file convergence_8x8_parameters you can find the parameters that will make all algorithms converge in the 8x8 map.

4. **Run the scripts** \
   To run the different scripts just do e.g.: ```python baseC51/c51.py``` or ```uv run baseC51/c51.py``` if you are using `uv` 
