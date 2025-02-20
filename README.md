## XAI Project C51 + Rules on MiniGrid

In this repository you can find the code to run the C51 algorithm and its variations using additional rules on the MiniGrid environment from Gymnasium. The code is adapted from CleanRL to work with MiniGrid.

### Repository Structure

- **C51, C51rexp, C51rt, C51rtv2, C51rtv3**:  
  In each of these folders you will find the results of a run for each model. Each folder contains:
  - An image with the model's performance.
  - A text file with the arguments (`args`) used for that run.
  - The trained model saved as a `.pt` file.
  - TensorBoard log directories with training/evaluation logs.
  
- **baseC51**:  
  Contains the implementation of the normal C51 algorithm without any additional rules.

- **rules_in_training**:  
  Contains the implementations where the rules are applied during training. The models in these folders include variations that modify the training process using predefined rules.

- **rules_in_eval**:  
  Contains the implementations where the rules are applied during evaluation. In these scripts the rules are applied on models trained with the base C51 algorithm.

- **config.py**:  
  This file is used to set the different parameters for each run (learning rate, environment settings, etc.).

### How to run

1. **Create a Python environment**  
   Create a virtual environment with Python 3.10.12 and install the packages. For example, using conda:
   ```bash
   conda create -n xai-project python=3.10.12
   conda activate xai-project
   pip install -r requirements.txt
   ```

   Or you can install `uv` and follow the next steps and it will automatically create a .venv folder with the proper virtual environment the first time you run a script.
2. **Configure wandb (optional)** \
   If you want to use wandb tracking, create a `.env` file in the repository root and set the proper environment variables (WANDB_PROJECT_NAME, WANDB_ENTITY).

3. **Configure the config.py file**

4. **Run the scripts** \
   To run the different scripts just do e.g.: ```python baseC51/c51.py``` or ```uv run baseC51/c51.py``` if you are using `uv` 
