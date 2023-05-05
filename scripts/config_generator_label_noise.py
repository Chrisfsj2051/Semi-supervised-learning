import yaml
import os

def handle(src_file):
    tar_file = src_file.replace('cifar10_40', 'cifar10_4000')
    # Load the YAML config file
    with open(src_file, 'r') as f:
        config = yaml.safe_load(f)

    # Modify the value of the 'num_labels' key
    config['num_labels'] = 4000
    config['save_name'] = config['save_name'].replace('cifar10_40', 'cifar10_4000')

    # Save the modified YAML config file
    with open(tar_file, 'w') as f:
        yaml.dump(config, f)

    print(tar_file)

for root, dirs, files in os.walk('config/vcc_ssl_full/noisy_cifar10'):
    for file in files:
        if '_40' in file:
            handle(os.path.join(root, file))
        # break