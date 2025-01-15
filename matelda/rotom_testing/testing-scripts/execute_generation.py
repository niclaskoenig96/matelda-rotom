import subprocess
import os

conda_env_name = "my_env"
python_script = "one_by_one_generate.py" 


try:
    # Activate conda environment and run the Python script
    command = f"conda run -n {conda_env_name} python {python_script}"
    subprocess.run(command, check=True, shell=True)
    print(f"Executed {python_script} in conda environment '{conda_env_name}' successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error occurred while executing script: {e}")


