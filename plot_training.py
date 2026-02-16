import matplotlib.pyplot as plt
import re
import argparse
import os
import json

def parse_log(log_path):
    steps = []
    rewards = []
    losses = []
    
    step_count = 0
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("{") and "}" in line:
                try:
                    # Parse as text to be robust
                    reward_match = re.search(r"'reward':\s*([0-9\.\-]+)", line)
                    loss_match = re.search(r"'loss':\s*([0-9\.\-]+)", line)
                    
                    if reward_match or loss_match:
                        step_count += 1
                        step = step_count
                        
                        step_match = re.search(r"'step':\s*(\d+)", line)
                        if step_match:
                            step = int(step_match.group(1))
                            
                        rewards.append(float(reward_match.group(1)) if reward_match else 0)
                        losses.append(float(loss_match.group(1)) if loss_match else 0)
                        steps.append(step)
                except Exception:
                    continue
                    
    return steps, rewards, losses

def plot_metrics(steps, rewards, losses, output_file):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Reward Plot
    color = 'tab:blue'
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(steps, rewards, color=color, label='Reward')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=0.5, color='r', linestyle='--', label='Unsafe Threshold (0.5)')
    ax1.set_title('Training Reward')
    ax1.legend()
    ax1.grid(True)

    # Loss Plot
    color = 'tab:orange'
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss', color=color)
    ax2.plot(steps, losses, color=color, label='Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_title('Training Loss')
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="training_log_oblit.txt")
    parser.add_argument("--output_file", type=str, default="training_trajectory.png")
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"Log file {args.log_file} not found.")
        return

    steps, rewards, losses = parse_log(args.log_file)
    
    if not steps:
        print("No metrics found in log file yet.")
        return
        
    plot_metrics(steps, rewards, losses, args.output_file)

if __name__ == "__main__":
    main()
