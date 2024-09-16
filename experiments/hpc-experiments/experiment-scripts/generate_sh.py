import os

project_home = "/home/thakarr/D-MO-Operator/experiments/hpc-experiments/"

def generate_bash_scripts(time, env, alg, num_scripts, output_dir):
    # Create the full path to the output directory inside the project_home
    full_output_dir = os.path.join(project_home, output_dir)
    
    # Ensure the entire directory structure exists or raise an error if it doesn't
    if not os.path.exists(full_output_dir):
        raise FileNotFoundError(f"The directory {full_output_dir} does not exist. Please create it before proceeding.")
    
    # Alternatively, to ensure the directory gets created (if allowed)
    # os.makedirs(full_output_dir, exist_ok=True)

    for i in range(1, num_scripts + 1):
        script_name = f"job_script_{env}_{alg}_{i}.sh"
        script_path = os.path.join(full_output_dir, script_name)
        
        with open(script_path, 'w') as file:
            file.write("#!/bin/bash\n")
            file.write(f"#SBATCH --time={time}\n")
            file.write("#SBATCH --constraint=skylake\n")
            file.write("#SBATCH --mem=16G\n")
            file.write("#SBATCH -c 8\n\n")
            file.write("module load conda\n\n")
            file.write(f'/home/thakarr/D-MO-Operator/build/MOD "{project_home + env}.yaml" "{project_home}data/" "{alg}"\n')

        # Make the file executable
        os.chmod(script_path, 0o755)

        print(f"Generated script: {script_path}")

# Example usage:
time = "0-12:00:00"  # Set the time
envs = ["MOBP-2objs-easy", "MOBP-2objs-hard", "MOREP-2objs-easy", "MOREP-2objs-hard"]
algs = ["mod", "mod_abl", "mod_teama_abl", "nsga"]
num_scripts = 10  # Number of bash scripts to generate
output_dir = "experiment-scripts/bash_scripts"  # Directory to save the scripts

for env in envs:
    for alg in algs:
        generate_bash_scripts(time, env, alg, num_scripts, output_dir)
