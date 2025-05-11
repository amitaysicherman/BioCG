from os.path import join as pjoin
from os.path import exists
from torch.utils.data import Dataset as TorchDataset
import torch
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

import random
import os
from typing import List, Set, Tuple, Any, Dict, Union  # Added for type hinting

import pandas as pd

# --- Configuration Constants ---
# These define the proportions for splitting the dataset.
TRAIN_SIZE: float = 0.85
VALID_SIZE: float = 0.05
TEST_SIZE: float = 0.1


# Input file path


# --- Class Definition ---
class Sample:
    """Represents a single data sample with SMILES, FASTA, and a label."""

    def __init__(self, smiles: str, fasta: str, label_data: Union[str, int, float]):
        self.smiles: str = smiles
        self.fasta: str = fasta
        # The original code used int(float(label)). This handles cases like "1.0" or "0.0".
        try:
            self.label: int = int(float(label_data))
        except ValueError as e:
            raise ValueError(
                f"Invalid label format for sample (smiles: {smiles[:30]}..., fasta: {fasta[:30]}...): {label_data}"
            ) from e

    def is_pos(self) -> bool:
        """Checks if the sample has a positive label (1)."""
        return self.label == 1

    def is_neg(self) -> bool:
        """Checks if the sample has a negative label (0)."""
        return self.label == 0


def _filter_samples_by_attribute_in_reference_set(
        samples_to_filter: List[Sample],
        reference_samples: List[Sample],
        attribute_name: str
) -> List[Sample]:
    """
    Filters 'samples_to_filter', keeping only those whose specified 'attribute_name'
    value is present in the 'reference_samples'.
    Returns a new list of filtered samples.
    (Original: remove_not_appear_indexes - modified to return new list for clarity)
    """
    reference_attribute_values = {getattr(sample, attribute_name) for sample in reference_samples}

    filtered_list = [
        sample for sample in samples_to_filter
        if getattr(sample, attribute_name) in reference_attribute_values
    ]
    return filtered_list


def _partition_samples_by_primary_attribute(
        all_samples: List[Sample],
        entities_for_holdout_set: Set[Any],
        primary_attribute_name: str
) -> Tuple[List[Sample], List[Sample]]:
    """
    Partitions 'all_samples' into two lists: a training set and a holdout set.
    The split is based on whether a sample's 'primary_attribute_name' value
    is present in 'entities_for_holdout_set'.
    (Original: process_samples)
    """
    train_set = []
    holdout_set = []
    for sample in all_samples:
        if getattr(sample, primary_attribute_name) in entities_for_holdout_set:
            holdout_set.append(sample)
        else:
            train_set.append(sample)
    return train_set, holdout_set


def _split_holdout_into_validation_and_test(
        holdout_samples: List[Sample],
        global_valid_proportion: float,
        global_test_proportion: float
) -> Tuple[List[Sample], List[Sample]]:
    """
    Splits the 'holdout_samples' into validation and test sets.
    Samples are shuffled before splitting. The proportions are relative to
    the combined size of validation and test sets.
    (Original: split_val_test - made robust to zero proportions)
    """
    if not holdout_samples:
        return [], []

    random.shuffle(holdout_samples)  # Shuffle the list in-place

    combined_holdout_proportion = global_valid_proportion + global_test_proportion

    if combined_holdout_proportion == 0:
        # If both global proportions are zero, no samples for validation. All remain for test (or none if list empty).
        # This also handles if holdout_samples is empty.
        num_validation_samples = 0
    else:
        # Calculate the fraction of holdout_samples that should go to the validation set
        # This ratio is equivalent to: global_valid_proportion / (global_valid_proportion + global_test_proportion)
        ratio_for_validation_in_holdout = global_valid_proportion / combined_holdout_proportion
        num_validation_samples = int(len(holdout_samples) * ratio_for_validation_in_holdout)

    validation_set = holdout_samples[:num_validation_samples]
    test_set = holdout_samples[num_validation_samples:]
    return validation_set, test_set


def load_and_preprocess_samples(csv_path: str) -> Tuple[List[Sample], List[Sample]]:
    """
    Loads samples from a CSV file, separates them into positive and negative,
    and filters negative samples based on criteria from positive samples.
    """
    try:
        dataframe = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return [], []
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return [], []

    all_loaded_samples = [
        Sample(row['SMILES'], row['Protein'], row["Y"])
        for _, row in dataframe.iterrows()
    ]

    positive_samples = [s for s in all_loaded_samples if s.is_pos()]
    negative_samples_initial = [s for s in all_loaded_samples if s.is_neg()]

    # Filter negative samples: keep only those whose SMILES and FASTA
    # are also found in the set of positive samples.
    positive_smiles_set = {sample.smiles for sample in positive_samples}
    positive_fasta_set = {sample.fasta for sample in positive_samples}

    count_neg_before_filter = len(negative_samples_initial)
    filtered_negative_samples = [
        s for s in negative_samples_initial
        if s.smiles in positive_smiles_set and s.fasta in positive_fasta_set
    ]
    random.shuffle(filtered_negative_samples)  # Shuffle for random distribution

    print(
        f"Negative samples count: {len(filtered_negative_samples)} (originally {count_neg_before_filter} before filtering based on positive set criteria)")

    return positive_samples, filtered_negative_samples


def get_dti_datasets(dataset, cold_fasta, cold_smiles, seed=42):
    random.seed(seed)
    input_csv_file = f"data/{dataset}.csv"
    positive_samples, negative_samples = load_and_preprocess_samples(input_csv_file)

    if not positive_samples and not negative_samples:
        print("No samples loaded. Exiting.")
        exit()

    num_total_positive = len(positive_samples)
    num_total_negative = len(negative_samples)

    # Initialize lists for each dataset split (positive and negative)
    train_pos_samples: List[Sample] = []
    valid_pos_samples: List[Sample] = []
    test_pos_samples: List[Sample] = []

    train_neg_samples: List[Sample] = []
    valid_neg_samples: List[Sample] = []
    test_neg_samples: List[Sample] = []

    # 2. Perform Data Splitting based on Configuration

    if not cold_fasta and not cold_smiles:
        # --- "Warm" Split (Standard/Mixed Split) ---
        print("\nPerforming warm split...")

        # For positive samples:
        # Preferentially assign samples with new SMILES or FASTA to the training set.
        seen_smiles_in_train_pool = set()
        seen_fasta_in_train_pool = set()

        initial_preferential_train_pos = []
        remaining_after_preference_pos = []

        # The order of positive_samples matters for which "new" items are picked first.
        # Original code did not shuffle positive_samples before this step.
        for sample in positive_samples:  # Iterating over positive samples
            is_new_smiles = sample.smiles not in seen_smiles_in_train_pool
            is_new_fasta = sample.fasta not in seen_fasta_in_train_pool

            if is_new_smiles or is_new_fasta:
                seen_smiles_in_train_pool.add(sample.smiles)
                seen_fasta_in_train_pool.add(sample.fasta)
                initial_preferential_train_pos.append(sample)
            else:
                remaining_after_preference_pos.append(sample)

        random.shuffle(remaining_after_preference_pos)  # Shuffle the remainder

        # Calculate target number of positive samples for training
        target_num_pos_train = int(num_total_positive * TRAIN_SIZE)

        # Start with preferentially selected positive samples
        train_pos_samples.extend(initial_preferential_train_pos)

        # Fill remaining training quota for positive samples
        num_additional_pos_train_needed = target_num_pos_train - len(train_pos_samples)

        split_idx_for_remaining = 0
        if num_additional_pos_train_needed > 0:
            train_pos_samples.extend(remaining_after_preference_pos[:num_additional_pos_train_needed])
            split_idx_for_remaining = num_additional_pos_train_needed

        # Assign to validation and test sets for positive samples from the rest
        target_num_pos_valid = int(num_total_positive * VALID_SIZE)
        valid_pos_samples.extend(
            remaining_after_preference_pos[split_idx_for_remaining: split_idx_for_remaining + target_num_pos_valid]
        )
        split_idx_for_remaining += target_num_pos_valid

        test_pos_samples.extend(remaining_after_preference_pos[split_idx_for_remaining:])

        # Split negative samples proportionally (they are already shuffled)
        neg_train_end_idx = int(num_total_negative * TRAIN_SIZE)
        neg_valid_end_idx = int(num_total_negative * (TRAIN_SIZE + VALID_SIZE))

        train_neg_samples = negative_samples[:neg_train_end_idx]
        valid_neg_samples = negative_samples[neg_train_end_idx:neg_valid_end_idx]
        test_neg_samples = negative_samples[neg_valid_end_idx:]

    else:
        # --- "Cold" Split (based on FASTA or SMILES) ---
        split_type = "FASTA" if cold_fasta else "SMILES"
        print(f"\nPerforming cold split by {split_type}...")

        primary_attr: str = "fasta" if cold_fasta else "smiles"
        secondary_attr: str = "smiles" if cold_fasta else "fasta"

        # Determine unique entities (FASTA or SMILES) from positive samples for splitting
        # Using positive samples to define entities for holdout ensures coldness relative to known positives.
        all_primary_entities_in_pos_samples: Set[str]
        if primary_attr == "fasta":
            all_primary_entities_in_pos_samples = {sample.fasta for sample in positive_samples}
        else:  # primary_attr == "smiles"
            all_primary_entities_in_pos_samples = {sample.smiles for sample in positive_samples}

        if not all_primary_entities_in_pos_samples:
            print(
                f"Warning: No unique {primary_attr} entities found in positive samples for cold split. Results may be skewed.")
            # Fallback or specific handling might be needed if this set is empty
            # For now, proceeding will likely result in empty holdout sets if num_entities_to_select is 0

        # Select entities that will define the validation/test sets (holdout entities)
        num_entities_for_holdout = int(len(all_primary_entities_in_pos_samples) * (VALID_SIZE + TEST_SIZE))

        # Ensure random.sample k is not greater than population size
        if num_entities_for_holdout > len(all_primary_entities_in_pos_samples):
            num_entities_for_holdout = len(all_primary_entities_in_pos_samples)

        # `random.sample` requires a sequence (list or tuple), not a set directly
        entities_for_holdout_set = set(
            random.sample(list(all_primary_entities_in_pos_samples), num_entities_for_holdout)
        )

        # Process Positive Samples for Cold Split
        train_pos_samples_temp, holdout_pos_samples_temp = _partition_samples_by_primary_attribute(
            positive_samples, entities_for_holdout_set, primary_attr
        )
        # Filter holdout set: secondary attribute must appear in the training set's secondary attributes
        # This ensures that the validation/test items are not completely novel in *both* dimensions if possible.
        holdout_pos_samples_filtered = _filter_samples_by_attribute_in_reference_set(
            holdout_pos_samples_temp, train_pos_samples_temp, secondary_attr
        )
        valid_pos_samples, test_pos_samples = _split_holdout_into_validation_and_test(
            holdout_pos_samples_filtered, VALID_SIZE, TEST_SIZE
        )
        train_pos_samples = train_pos_samples_temp  # Final positive training set

        # Process Negative Samples for Cold Split
        # Negative samples are also split by the SAME primary attribute entities selected from positive samples
        train_neg_samples_temp, holdout_neg_samples_temp = _partition_samples_by_primary_attribute(
            negative_samples, entities_for_holdout_set, primary_attr
        )
        # Filter holdout negative samples based on secondary attributes present in POSITIVE training samples
        holdout_neg_samples_filtered = _filter_samples_by_attribute_in_reference_set(
            holdout_neg_samples_temp, train_pos_samples, secondary_attr  # reference is train_pos_samples
        )
        valid_neg_samples, test_neg_samples = _split_holdout_into_validation_and_test(
            holdout_neg_samples_filtered, VALID_SIZE, TEST_SIZE
        )
        train_neg_samples = train_neg_samples_temp  # Final negative training set

    # 3. Output Information about the Splits
    # The original code printed "Saving {name} samples" and extracted SMILES/FASTA,
    # but didn't write to files within the provided snippet. This section replicates that informational output.

    print("\n--- DATASET SPLIT SUMMARY ---")
    dataset_splits_info: Dict[str, Tuple[List[Sample], List[Sample]]] = {
        "train": (train_pos_samples, train_neg_samples),
        "valid": (valid_pos_samples, valid_neg_samples),
        "test": (test_pos_samples, test_neg_samples),
    }

    for split_name, (pos_samples, neg_samples) in dataset_splits_info.items():
        print(f"{split_name} - Positive: {len(pos_samples)}, Negative: {len(neg_samples)}")
    print("\n--- END OF SPLIT SUMMARY ---")

    train_pos_src = [sample.smiles for sample in train_pos_samples]
    train_pos_tgt = [sample.fasta for sample in train_pos_samples]
    train_neg_src = [sample.smiles for sample in train_neg_samples]
    train_neg_tgt = [sample.fasta for sample in train_neg_samples]
    valid_pos_src = [sample.smiles for sample in valid_pos_samples]
    valid_pos_tgt = [sample.fasta for sample in valid_pos_samples]
    valid_neg_src = [sample.smiles for sample in valid_neg_samples]
    valid_neg_tgt = [sample.fasta for sample in valid_neg_samples]
    test_pos_src = [sample.smiles for sample in test_pos_samples]
    test_pos_tgt = [sample.fasta for sample in test_pos_samples]
    test_neg_src = [sample.smiles for sample in test_neg_samples]
    test_neg_tgt = [sample.fasta for sample in test_neg_samples]
    return train_pos_src, train_pos_tgt, train_neg_src, train_neg_tgt, valid_pos_src, valid_pos_tgt, valid_neg_src, valid_neg_tgt, test_pos_src, test_pos_tgt, test_neg_src, test_neg_tgt


def get_care_datasets():
    with open("data/care/train_reaction.txt", "r") as f:
        train_pos_src = f.read().splitlines()
    with open("data/care/train_enzyme.txt", "r") as f:
        train_pos_tgt = f.read().splitlines()
    with open("data/care/test_reaction.txt", "r") as f:
        test_pos_src = f.read().splitlines()
    with open("data/care/test_enzyme.txt", "r") as f:
        test_pos_tgt = f.read().splitlines()
    return train_pos_src, train_pos_tgt, test_pos_src, test_pos_tgt

class SrcTgtDataset(TorchDataset):
    def __init__(self, src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, src_encoder=None, max_length=256,
                 pooling=False, train_encoder=False):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.max_length = max_length
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_encoder = src_encoder  # This can be None if we use end-to-end model
        self.pooling = pooling
        self.train_encoder = train_encoder
        self.memory = {}

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        # Pre-encoded version (used when encoder is frozen)
        if not self.train_encoder and self.src_encoder is not None and src_text in self.memory:
            src_encoder_outputs, src_attention_mask = self.memory[src_text]
        # When training encoder or first time processing
        else:
            src_tokens = self.src_tokenizer(
                src_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
            )

            # If we're not training encoder, we pre-compute encoder outputs
            if not self.train_encoder and self.src_encoder is not None:
                src_tokens = {k: v.to(device) for k, v in src_tokens.items()}
                with torch.no_grad():
                    src_encoder_outputs = self.src_encoder(**src_tokens)
                if self.pooling:
                    if hasattr(self.src_encoder, "pooler_output"):
                        src_encoder_outputs = src_encoder_outputs.pooler_output
                    else:
                        src_encoder_outputs = src_encoder_outputs.last_hidden_state.mean(dim=1)
                    src_attention_mask = torch.ones(1).to(device)
                else:
                    src_encoder_outputs = src_encoder_outputs.last_hidden_state.squeeze(0)
                    src_attention_mask = src_tokens["attention_mask"].squeeze(0)
                src_encoder_outputs = src_encoder_outputs.detach().cpu()
                src_attention_mask = src_attention_mask.detach().cpu()

                # Save to memory if we pre-compute
                self.memory[src_text] = (src_encoder_outputs, src_attention_mask)
            # When training encoder, we just return the input tokens
            else:
                src_encoder_outputs = None
                src_attention_mask = src_tokens["attention_mask"].squeeze(0)
                # We need to keep input_tokens for the model
                src_input_ids = src_tokens["input_ids"].squeeze(0)

        tgt_tokens = self.tgt_tokenizer(
            tgt_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        labels = tgt_tokens["input_ids"].clone()
        labels[labels == self.tgt_tokenizer.pad_token_id] = -100

        # Return format depends on whether we're training the encoder
        if not self.train_encoder and self.src_encoder is not None:
            return dict(
                encoder_outputs=src_encoder_outputs,
                encoder_attention_mask=src_attention_mask,
                input_ids=tgt_tokens["input_ids"].squeeze(0),
                attention_mask=tgt_tokens["attention_mask"].squeeze(0),
                labels=labels.squeeze(0),
            )
        else:
            return dict(
                src_input_ids=src_input_ids,
                src_attention_mask=src_tokens["attention_mask"].squeeze(0),
                input_ids=tgt_tokens["input_ids"].squeeze(0),
                attention_mask=tgt_tokens["attention_mask"].squeeze(0),
                labels=labels.squeeze(0),
            )


if __name__ == "__main__":
    # Example usage
    input_csv_file = "biosnap"  # Replace with your actual CSV file path
    cold_fasta = True  # Set to True for cold split by FASTA
    cold_smiles = False  # Set to True for cold split by SMILES

    train_pos_src, train_pos_tgt, train_neg_src, train_neg_tgt, valid_pos_src, valid_pos_tgt, valid_neg_src, valid_neg_tgt, test_pos_src, test_pos_tgt, test_neg_src, test_neg_tgt = get_dti_datasets(input_csv_file, cold_fasta, cold_smiles)
    # Example of how to use the dataset
    a = 2
