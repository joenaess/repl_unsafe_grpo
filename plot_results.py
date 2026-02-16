import matplotlib.pyplot as plt
import re
import argparse
import os

def parse_log(log_path):
    steps = []
    rewards = []
    losses = []
    
    # Regex for standard HF logging
    # Example: {'loss': 0.123, 'reward': 0.456, 'step': 10}
    # Or text format: "Step: 10, Loss: 0.123, Reward: 0.456"
    # GRPOTrainer might log specific keys.
    
    step_count = 0
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("{") and "}" in line:
                try:
                    # Parse as simple text search to avoid json issues
                    reward_match = re.search(r"'reward':\s*([0-9\.\-]+)", line)
                    loss_match = re.search(r"'loss':\s*([0-9\.\-]+)", line)
                    
                    if reward_match or loss_match:
                        step_count += 1
                        step = step_count
                        
                        # Try to find explicit step if available
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
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(steps, rewards, color=color, label='Reward')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.1, 1.1)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(steps, losses, color=color, linestyle='--', label='Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Training Progress: Reward & Loss')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid(True)
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="training_log_oblit.txt")
    parser.add_argument("--output_file", type=str, default="training_plot.png")
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
