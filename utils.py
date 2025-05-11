import os
import csv
import torch
import numpy as np
from sklearn.cluster import KMeans
from transformers import TrainerCallback


class EvalLoggingCallback(TrainerCallback):
    """
    Callback to log evaluation metrics to a CSV file during training.
    """

    def __init__(self, output_dir, filename="eval_logs.csv"):
        """
        Initialize the callback.

        Args:
            output_dir: Directory where the CSV file will be saved
            filename: Name of the CSV file
        """
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, filename)
        self.headers = None
        self.csv_file = None
        self.writer = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Called after evaluation. Logs metrics to CSV file.
        """
        if metrics is None:
            return

        # Get current step
        step = state.global_step

        # Add step to metrics
        metrics['step'] = step

        # Check if directory exists, create if not
        os.makedirs(self.output_dir, exist_ok=True)

        # If file doesn't exist, create and write headers
        if not os.path.exists(self.csv_path):
            # Create CSV file with headers
            with open(self.csv_path, 'w', newline='') as f:
                # Get all keys from metrics as headers
                self.headers = sorted(metrics.keys())
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
                writer.writerow(metrics)
        else:
            # Append to existing file
            with open(self.csv_path, 'a', newline='') as f:
                # Read existing headers if needed
                if self.headers is None:
                    with open(self.csv_path, 'r', newline='') as read_f:
                        reader = csv.reader(read_f)
                        self.headers = next(reader)  # Get headers from first row

                # Check if we have new metrics that weren't in original headers
                new_headers = [h for h in metrics.keys() if h not in self.headers]

                if new_headers:  # If we have new headers, we need to rewrite the file
                    self.headers.extend(new_headers)
                    # Read existing data
                    with open(self.csv_path, 'r', newline='') as read_f:
                        reader = csv.DictReader(read_f)
                        existing_data = list(reader)

                    # Write all data with updated headers
                    with open(self.csv_path, 'w', newline='') as write_f:
                        writer = csv.DictWriter(write_f, fieldnames=self.headers)
                        writer.writeheader()
                        for row in existing_data:
                            writer.writerow(row)
                        writer.writerow(metrics)
                else:
                    # Just append to existing file with consistent headers
                    writer = csv.DictWriter(f, fieldnames=self.headers)
                    writer.writerow(metrics)

        print(f"Evaluation metrics at step {step} logged to {self.csv_path}")


def get_config():
    """
    Get configuration from command line arguments.
    """
    import argparse
    import yaml  # For loading the config file

    parser = argparse.ArgumentParser(description="Train Encoder-Decoder Model with Configuration")
    parser.add_argument("--config", type=str, default="configs/biosnap_seen.yaml",
                        help="Path to the YAML configuration file")
    cli_args = parser.parse_args()

    # Load configuration from YAML file
    with open(cli_args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Override with command line arguments if provided
    return config


def get_random_tokens(tokenizer, num_tokens=10):
    """Get random tokens from the tokenizer"""

    vocab_size = len(tokenizer)
    random_tokens = np.random.randint(0, vocab_size, num_tokens)
    return random_tokens


class RamdomReplace:
    def __init__(self, tokenizer, num_tokens=256, vocab_size=0):
        self.tokenizer = tokenizer
        if vocab_size == 0:
            vocab_size = len(tokenizer)
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens
        self.memory = {}

    def get_random_tokens(self, text):
        if text in self.memory:
            return self.memory[text]
        random_tokens = np.random.randint(0, self.vocab_size, self.num_tokens)
        random_text = self.tokenizer.decode(random_tokens)
        self.memory[text] = random_text
        return random_text
