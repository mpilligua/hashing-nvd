import os
import subprocess
import time

def get_available_gpus():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], stdout=subprocess.PIPE)
        gpus = result.stdout.decode('utf-8').strip().split('\n')
        return [int(gpu) for gpu in gpus]
    except Exception as e:
        print(f"Error fetching GPU information: {e}")
        return []

def is_gpu_busy(gpu_id):
    try:
        command = f"nvidia-smi --id={gpu_id} --query-compute-apps=pid --format=csv,noheader"
        result = subprocess.run(command.split(), stdout=subprocess.PIPE)
        processes = result.stdout.decode('utf-8').strip().split('\n')
        return len(processes) > 0 and processes[0] != ''
    except Exception as e:
        print(f"Error checking GPU status: {e}")
        return True

def wait_for_free_gpu(gpus):
    while True:
        for gpu_id in gpus:
            if not is_gpu_busy(gpu_id):
                return gpu_id
        time.sleep(5)

def create_config_file(base_config_path, new_video_name):
    # Read the base configuration file
    with open(base_config_path, 'r') as file:
        config_content = file.read()
    
    # Replace all instances of 'blackswan' with the new video name
    new_config_content = config_content.replace('blackswan', new_video_name)
    
    # Define the new configuration file name
    new_config_file = f"/data/users/mpilligua/hashing-nvd/config/chg_enc/{new_video_name}.py"
    os.makedirs(os.path.dirname(new_config_file), exist_ok=True)
    
    # Save the new configuration file
    with open(new_config_file, 'w') as file:
        file.write(new_config_content)
    
    return new_config_file

def setup_environment():
    os.chdir('/data/users/mpilligua/hashing-nvd')
    os.environ['LD_LIBRARY_PATH'] += ':/home/mpilligua/anaconda3/envs/hashing_nvd2/lib'
    subprocess.run('source activate hashing_nvd2', shell=True, executable="/bin/bash")

def train_on_gpus(config_files):
    setup_environment()  # Set up the environment before training
    
    gpus = get_available_gpus()
    gpus = [gpu for gpu in gpus if gpu not in gpus_to_avoid]
    print("Available GPUs:", gpus)

    if not gpus:
        print("No GPUs found!")
        return

    os.chdir('/data/users/mpilligua/hashing-nvd/')

    for config_file in config_files:
        gpu_id = wait_for_free_gpu(gpus)
        print("Using GPU:", gpu_id)
        
        command = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py {config_file} > {config_file}.log 2>&1 &'
        print(f"Running command: {command}")
        os.system(command)
        time.sleep(60)

if __name__ == "__main__":
    gpus_to_avoid = []  # Replace with the GPUs you want to avoid
    base_config_file = '/data/users/mpilligua/hashing-nvd/config/config_blackswan_chg_enc.py'  # Replace with the actual path to your base config file
    
    # List of new video names you want to create config files for
    video_names = ['swing', 'train', 'surf', 'stroller', 'soccerball', 'rollerblade', 'paragliding-launch', 'car-turn', 'boat', 'bus']  # Replace with your actual video names
    
    # Create new config files based on the base config file and video names
    config_files = []
    for video_name in video_names:
        config_file = create_config_file(base_config_file, video_name)
        config_files.append(config_file)
    
    # Start training on GPUs
    train_on_gpus(config_files)
