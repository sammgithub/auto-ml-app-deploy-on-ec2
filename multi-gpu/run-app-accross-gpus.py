import subprocess
import time

# Configuration
instance_type = 'g4dn.xlarge'
ami_id = 'ami-0zzzzzzzzzzzzx' #update
key_name = 'xpem'
key_file_path = '/your/pem/file/path/xpem.pem' #update
security_group_ids = 'sg-0xxxxxxxxxxxxx' #update
subnet_id = 'subnet-5xxxxxxx' #update 
vpc_id = 'vpc-814xxxx' #update

# Get the number of processes per node from the command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nproc_per_node', type=int, required=True, help='Number of processes per node')
args = parser.parse_args()
nproc_per_node = args.nproc_per_node

# Launch EC2 instance
def launch_instance():
    command = [
        "aws", "ec2", "run-instances",
        "--image-id", ami_id,
        "--count", "1",
        "--instance-type", instance_type,
        "--key-name", key_name,
        "--security-group-ids", security_group_ids,
        "--subnet-id", subnet_id,
        "--query", "Instances[0].InstanceId",
        "--output", "text"
    ]
    instance_id = subprocess.check_output(command).decode('utf-8').strip()
    print(f"Waiting for instance {instance_id} to start...")
    return instance_id

# Wait for the instance to be running and get its public IP
def get_instance_ip(instance_id):
    while True:
        command = [
            "aws", "ec2", "describe-instances",
            "--instance-ids", instance_id,
            "--query", "Reservations[0].Instances[0].PublicIpAddress",
            "--output", "text"
        ]
        instance_ip = subprocess.check_output(command).decode('utf-8').strip()
        if instance_ip != 'None':
            return instance_ip
        time.sleep(5)

# Install PyTorch and run the script
def run_commands(instance_ip):
    # Copy the ddp.py script to the instance
    scp_command = f"scp -i {key_file_path} -o StrictHostKeyChecking=no ddp.py ec2-user@{instance_ip}:/home/ec2-user/ddp.py"
    subprocess.run(scp_command, shell=True)

    # SSH into the instance and run commands
    ssh_command = f"yes yes | ssh -i {key_file_path} -o StrictHostKeyChecking=no ec2-user@{instance_ip} << 'EOF'\n"
    commands = [
        "pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117",
        "nvidia-smi",  # Check GPU status
        f"torchrun --nproc_per_node={nproc_per_node} /home/ec2-user/ddp.py"
    ]
    for command in commands:
        ssh_command += command + "\n"
    ssh_command += "EOF"
    
    subprocess.run(ssh_command, shell=True)

# Terminate the instance
def terminate_instance(instance_id):
    command = ["aws", "ec2", "terminate-instances", "--instance-ids", instance_id]
    subprocess.run(command)

if __name__ == "__main__":
    instance_id = launch_instance()
    instance_ip = get_instance_ip(instance_id)
    run_commands(instance_ip)
    terminate_instance(instance_id)
