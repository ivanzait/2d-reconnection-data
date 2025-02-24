import os
import argparse
import random
import numpy as np
from constants import runs
import re


def extract_number(file_name):
    match = re.search(r'(\d+)', file_name)
    if match:
        return int(match.group(0))

def generate_sequences(steps, num_steps, step_size, exclude_steps):
    sequences = []
    for i in range(len(steps) - num_steps * step_size + 1):
        sequence = [steps[i + j * step_size] for j in range(num_steps)]
        if not set(sequence).intersection(exclude_steps) and all(step in steps for step in sequence):
            sequences.append(sequence)
    return sequences

def create_datasets(data_dir, out_dir, step_size, num_steps, num_test_sequences=1):
    for run in runs:
        run_id = run['id']
        t_min = run['t_min']
        t_max = run['t_max']

        os.makedirs(os.path.join(out_dir, 'test'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'train'), exist_ok=True)

        files = sorted(
            [f for f in os.listdir(data_dir) if f.startswith(run_id) and f.endswith('.npy')],
            key=extract_number)
        
        all_steps = [extract_number(f) for f in files]

        test_steps_all = []
        mid_point = (t_max + t_min) // 2
        max_possible_start = t_max - 32 * step_size + step_size
        test_start_options = list(range(mid_point, max_possible_start + 1, step_size))
        for _ in range(num_test_sequences):
            test_start_step = random.choice(test_start_options)
            test_start_options.remove(test_start_step)  # Ensure unique test sequences
            test_steps = list(range(test_start_step, test_start_step + 32 * step_size, step_size))
            test_steps_all.extend(test_steps)
            test_files = [files[step - t_min] for step in test_steps]
            test_data = np.stack([np.load(os.path.join(data_dir, f)) for f in test_files], axis=0)
            np.save(os.path.join(out_dir, 'test', f'{run_id}_{test_start_step}.npy'), test_data) 

        # Generate sequences excluding test steps
        valid_train_val_steps = sorted(set(all_steps) - set(test_steps_all))
        sequences = generate_sequences(valid_train_val_steps, num_steps, step_size, test_steps_all)

        # Split sequences into train and test
        random.shuffle(sequences)
        val_sample_size = int(len(sequences) * 0.1)
        val_sequences = sorted(sequences[:val_sample_size])
        train_sequences = sorted(sequences[val_sample_size:])

        # Process and save
        print(run_id, 'train', len(train_sequences))
        for seq in train_sequences:
            data = np.stack([np.load(os.path.join(data_dir, f"{run_id}_{step}.npy")) for step in seq], axis=0)
            np.save(os.path.join(out_dir, 'train', f'{run_id}_{seq[0]}.npy'), data)

        print(run_id, 'val', len(val_sequences))
        for seq in val_sequences:
            data = np.stack([np.load(os.path.join(data_dir, f"{run_id}_{step}.npy")) for step in seq], axis=0)
            np.save(os.path.join(out_dir, 'val', f'{run_id}_{seq[0]}.npy'), data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', type=str)
    parser.add_argument('-d', '--data_dir', type=str, default='frames')
    parser.add_argument('-s', '--step_size', type=int, default=5)
    parser.add_argument('-n', '--num_steps', type=int, default=500)
    parser.add_argument('-t', '--num_test_sequences', type=int, default=20)
    args = parser.parse_args()

    create_datasets(args.data_dir, args.out_dir, args.step_size, args.num_steps, args.num_test_sequences)
