import torch
from tqdm import tqdm
from typing import List, Dict, Any, Optional

TrieNodeStructure = Dict[int, Any]


class Trie:
    """
    A Trie data structure for storing sequences of tokens.

    Attributes:
        root (TrieNodeStructure): The root node of the trie, represented as a dictionary.
        pad_token_id (int): Token ID used for padding or as an empty/sentinel token.
        nodes_count (int): Total number of nodes in the trie (excluding the root).
        total_sequences (int): Total number of sequences (paths) inserted into the trie.
    """

    def __init__(self, token_sequences: List[List[int]], pad_token_id: int = 0):
        """
        Initializes the Trie and populates it with the given token sequences.

        Args:
            token_sequences (List[List[int]]): A list of sequences, where each sequence
                                               is a list of token IDs.
            pad_token_id (int, optional): The token ID to be treated as padding or
                                          an empty/sentinel value. Defaults to 0.
        """
        self.root: TrieNodeStructure = {}
        self.pad_token_id: int = pad_token_id
        self.nodes_count: int = 0
        self.total_sequences: int = 0

        pbar = tqdm(token_sequences, desc="Building Trie")
        for sequence in pbar:
            self._insert(sequence)
            self.total_sequences += 1
            pbar.set_description(f"Nodes: {self.nodes_count}, Sequences: {self.total_sequences}")

    def _insert(self, sequence: List[int]) -> None:
        """
        Inserts a single token sequence into the trie.

        Args:
            sequence (List[int]): A list of token IDs representing the sequence.
        """
        current_node = self.root
        for token_id in sequence:
            if token_id not in current_node:
                current_node[token_id] = {}
                self.nodes_count += 1
            current_node = current_node[token_id]

    def search_valid_next_tokens(self, prefix: List[int]) -> List[int]:
        """
        Searches the trie for a given prefix and returns a list of valid next tokens.

        Args:
            prefix (List[int]): A list of token IDs representing the prefix.

        Returns:
            List[int]: A list of token IDs that can follow the given prefix.
                       Returns `[self.pad_token_id]` if the prefix is not found
                       or if it leads to a terminal node with no further tokens.
        """
        current_node = self.root
        for token_id in prefix:
            if token_id not in current_node:
                return [self.pad_token_id]  # Prefix not found
            current_node = current_node[token_id]

        if not current_node:  # Prefix leads to a terminal node (empty dict)
            return [self.pad_token_id]
        return list(current_node.keys())

def build_trie_from_text(
        corpus: List[str],
        tokenizer: Any,
        max_sequence_length: int = 256,
        eos_token_id_override: Optional[int] = None
) -> Trie:
    """
    Builds a Trie from a list of text sentences/words using a provided tokenizer.

    Args:
        corpus (List[str]): A list of strings to build the trie from.
        tokenizer (Any): A tokenizer instance. Must have an `encode` method,
                         and attributes like `eos_token_id` and `pad_token_id`.
        max_sequence_length (int, optional): Maximum length for tokenized sequences.
                                             Defaults to 256.
        eos_token_id_override (Optional[int], optional): Override for EOS token ID.
                                                         Defaults to None.

    Returns:
        Trie: An initialized Trie object.
    """
    token_sequences: List[List[int]] = []

    eos_id = eos_token_id_override if eos_token_id_override is not None else getattr(tokenizer, 'eos_token_id', None)
    if eos_id is None:
        eos_id = getattr(tokenizer, 'vocab', {}).get("<eos>") or getattr(tokenizer, 'vocab', {}).get("</s>")
        if eos_id is None:
            raise ValueError("EOS token ID could not be determined. Provide eos_token_id_override.")

    pad_id = getattr(tokenizer, 'pad_token_id', 0)

    for text_item in tqdm(corpus, desc="Tokenizing for Trie"):
        tokens = tokenizer.encode(text_item, add_special_tokens=True)

        if len(tokens) > max_sequence_length:
            tokens = tokens[:max_sequence_length - 1] + [eos_id]
        # Assuming add_special_tokens=True handles EOS for non-truncated sequences appropriately.
        # If not, explicit EOS addition for non-truncated might be needed.

        token_sequences.append(tokens)

    return Trie(token_sequences, pad_token_id=pad_id)


def build_mask_from_trie(
        trie: Trie,
        sequences: torch.Tensor,  # Shape: (batch_size, seq_length)
        vocab_size: int
) -> torch.Tensor:
    """
    Generates a mask tensor indicating valid next tokens based on a trie and input sequences.

    The mask `M` is such that `M[batch, seq_pos, token_id] = 1` if `token_id` is a
    valid token to appear at `seq_pos + 1` given the prefix `sequences[batch, :seq_pos+1]`.

    Args:
        trie (Trie): The trie structure.
        sequences (torch.Tensor): Tensor of shape (batch_size, seq_length) containing token indices.
        vocab_size (int): The size of the vocabulary.

    Returns:
        torch.Tensor: A mask tensor of shape (batch_size, seq_length, vocab_size)
                      with 1.0s indicating valid next tokens.
    """
    batch_size, seq_length = sequences.shape
    device = sequences.device

    mask = torch.zeros((batch_size, seq_length, vocab_size), dtype=torch.float32, device=device)

    for b_idx in range(batch_size):
        current_trie_node: TrieNodeStructure = trie.root

        for s_idx in range(seq_length):
            token_at_s_idx = int(sequences[b_idx, s_idx].item())

            if token_at_s_idx == trie.pad_token_id:
                break

            if isinstance(current_trie_node, dict) and token_at_s_idx in current_trie_node:
                node_after_token_at_s_idx = current_trie_node[token_at_s_idx]

                if isinstance(node_after_token_at_s_idx, dict):
                    for child_token_id in node_after_token_at_s_idx:
                        if 0 <= child_token_id < vocab_size:
                            mask[b_idx, s_idx, child_token_id] = 1.0

                current_trie_node = node_after_token_at_s_idx
            else:
                break
    return mask
