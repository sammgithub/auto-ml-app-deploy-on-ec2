import boto3
import time
import paramiko
import datetime

def create_ec2_instance(image_id, instance_type, key_name, security_group_ids, subnet_id, count=1):
    ec2 = boto3.client('ec2')

    response = ec2.run_instances(
        ImageId=image_id,
        InstanceType=instance_type,
        KeyName=key_name,
        SecurityGroupIds=[security_group_ids],
        SubnetId=subnet_id,
        MinCount=count,
        MaxCount=count
    )

    instance_id = response['Instances'][0]['InstanceId']
    print(f"Instance {instance_id} creation initiated.")

    return instance_id

def wait_until_running(instance_id):
    ec2 = boto3.resource('ec2')

    instance = ec2.Instance(instance_id)
    print(f"Waiting for instance {instance_id} to be in running state...")

    instance.wait_until_running()
    print(f"Instance {instance_id} is now running.")

    # Adding a delay to ensure the instance is fully initialized
    time.sleep(60)

def get_instance_public_ip(instance_id):
    ec2 = boto3.resource('ec2')
    instance = ec2.Instance(instance_id)
    return instance.public_ip_address

def ssh_execute_command(ip, key_file_path, command):
    k = paramiko.RSAKey(filename=key_file_path)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print(f"Connecting to {ip}...")
    c.connect(hostname=ip, username="ec2-user", pkey=k)
    print("Connected")

    stdin, stdout, stderr = c.exec_command(command)
    output = stdout.read().decode()
    error = stderr.read().decode()

    c.close()

    return output, error

def install_docker(instance_ip, key_file_path):
    print(f"Installing Docker on instance at {instance_ip}...")

    commands = [
        "sudo yum install -y docker",
        "sudo systemctl start docker",
        "sudo usermod -aG docker ec2-user"  # Optional: Add ec2-user to the Docker group
    ]

    for command in commands:
        output, error = ssh_execute_command(instance_ip, key_file_path, command)
        if error:
            print(f"Failed to run command '{command}'.")
            print("Error:")
            print(error)
        else:
            print(f"Command '{command}' ran successfully.")
            print("Output:")
            print(output)

def upload_to_s3(bucket_name, key, content):
    s3 = boto3.client('s3')
    s3.put_object(Bucket=bucket_name, Key=key, Body=content)
    print(f"Output uploaded to S3 bucket '{bucket_name}' with key '{key}'")

def run_docker_container(instance_ip, key_file_path):
    print(f"Running Docker container on instance at {instance_ip}...")

    pull_command = "sudo docker pull sammbd/cot-app"
    run_command = "sudo docker run sammbd/cot-app"
    
    # First, pull the latest image
    output, error = ssh_execute_command(instance_ip, key_file_path, pull_command)
    if error:
        print("Failed to pull Docker image.")
        print("Error:")
        print(error)
    else:
        print("Docker image pulled successfully.")
        print("Output:")
        print(output)
    
    # Then, run the container and capture its output
    output, error = ssh_execute_command(instance_ip, key_file_path, run_command)
    if error:
        print("Failed to run Docker container.")
        print("Error:")
        print(error)
    else:
        print("Docker container ran successfully.")
        print("Output:")
        print(output)

        # Save output to S3
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        bucket_name = 'seraj-s3-samcli'
        key = f'cot-app-output-{timestamp}.txt'
        upload_to_s3(bucket_name, key, output)

if __name__ == "__main__":
    image_id = 'ami-089313d40efd067a9'
    instance_type = 't2.micro'
    key_name = 'apr24'
    security_group_ids = 'sg-02524143560b47240'
    subnet_id = 'subnet-5e38fc14'
    key_file_path = '/Users/s172/Documents/ec2/apr24.pem'

    instance_id = create_ec2_instance(image_id, instance_type, key_name, security_group_ids, subnet_id)
    wait_until_running(instance_id)
    
    instance_ip = get_instance_public_ip(instance_id)
    install_docker(instance_ip, key_file_path)
    run_docker_container(instance_ip, key_file_path)
