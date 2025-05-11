"""
Residual Vector Quantization (RVQ) for encoding high-dimensional data,
such as text or molecular embeddings, into discrete codes.

This script provides functionalities to:
1. Load pre-trained models (e.g., BERT, MoLFormer, ESM) to generate embeddings.
2. Train an RVQ model on these embeddings.
3. Convert input data (text/molecules/proteins) into sequences of discrete codes.
4. Apply these codes to transform datasets.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualVectorQuantizer:
    """
    Residual Vector Quantizer (RVQ).

    This class implements RVQ, a multi-stage vector quantization method.
    In each stage, a K-Means model is fitted to the residuals from the previous
    stage. The final quantized representation is a sequence of cluster indices
    from each stage.

    The process continues until each input vector has a unique quantized code
    or a stagnation condition (no improvement in unique codes) is met.
    """

    def __init__(
            self,
            model,
            tokenizer,
            n_clusters: int = 15,
            max_layers: int = 100,
            max_stagnation_steps: int = 5,
            noise_factor: float = 1.0 / 3.0,
            kmeans_kwargs: Optional[Dict[str, Any]] = None,
            random_state: Optional[int] = 42,
            random_fit: Optional[bool] = False,
    ):
        """
        Initialize the ResidualVectorQuantizer.

        Args:
            model : Pre-trained model for generating embeddings.
            tokenizer : Pre-trained tokenizer for the model.
            n_clusters (int): Number of clusters for K-Means in each layer.
            max_layers (int): Maximum number of quantization layers to create.
            max_stagnation_steps (int): Number of consecutive layers with no
                                        improvement in unique codes before
                                        attempting to add noise or stop.
            noise_factor (float): Factor to determine the standard deviation of
                                  random noise added to residuals during stagnation
                                  (std_dev = noise_factor * np.std(residual)).
            kmeans_kwargs (Optional[dict]): Additional keyword arguments for
                                            sklearn.cluster.KMeans.
            random_state (Optional[int]): Random state for K-Means for reproducibility.
        """
        if n_clusters <= 1:
            raise ValueError("n_clusters must be greater than 1.")

        self.n_clusters = n_clusters
        self.max_layers = max_layers
        self.max_stagnation_steps = max_stagnation_steps
        self.noise_factor = noise_factor
        self.kmeans_kwargs = kmeans_kwargs or {}
        self.random_state = random_state

        self.quantizers_: List[KMeans] = []
        self.n_layers_fitted_: int = 0
        self.is_fitted_: bool = False
        self.text_to_code_map: Dict[str, str] = {}
        self.model = model
        self.model.to(device)
        self.tokenizer = tokenizer
        self.line_to_code: Dict[str, str] = {}
        self.random_fit = random_fit

    def _convert_labels_to_comparable_codes(
            self, layered_labels: List[np.ndarray]
    ) -> List[Tuple[int, ...]]:
        """Converts a list of label arrays (one per layer) to a list of tuples for unique checking."""
        if not layered_labels:
            return []
        num_samples = len(layered_labels[0])
        num_layers = len(layered_labels)
        comparable_codes = [tuple(layered_labels[layer][sample_idx] for layer in range(num_layers))
                            for sample_idx in range(num_samples)]
        return comparable_codes

    def generate_embeddings(
            self,
            lines: List[str],
            batch_size: int = 32,
            max_length: int = 256,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of input strings (text, molecules, proteins).

        Args:
            lines (List[str]): List of input strings.
        Returns:
            np.ndarray: A numpy array of embeddings.
        """
        logger.info(f"Generating embeddings for {len(lines)} lines with batch size {batch_size}.")
        all_embeddings: List[np.ndarray] = []

        with torch.no_grad():  # Ensure no gradients are computed
            for i in tqdm(range(0, len(lines), batch_size), desc="Generating Embeddings"):
                batch_lines = lines[i: i + batch_size]
                tokens = self.tokenizer(
                    batch_lines,
                    return_tensors="pt",
                    padding=True,  # Pad to longest in batch
                    truncation=True,
                    max_length=max_length,
                )
                tokens = {k: v.to(device) for k, v in tokens.items()}
                outputs = self.model(**tokens)

                if 'attention_mask' in tokens:
                    attention_mask = tokens['attention_mask']
                    mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # Avoid division by zero
                    batch_emb = (sum_embeddings / sum_mask).detach().cpu().numpy()
                else:  # Fallback if no attention mask (less accurate for padded sequences)
                    batch_emb = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

                all_embeddings.append(batch_emb)
        embeddings_np = np.vstack(all_embeddings)
        logger.info(f"Generated embeddings of shape: {embeddings_np.shape}")
        return embeddings_np

    def create_line_to_code_map(
            self, lines: List[str], codes: np.ndarray
    ) -> Dict[str, str]:
        """
        Create a mapping from input lines to their string-represented RVQ codes.

        Args:
            lines (List[str]): The original input lines.
            codes (np.ndarray): The RVQ codes corresponding to the lines.

        Returns:
            Dict[str, str]: A dictionary mapping lines to space-separated code strings.
        """
        if len(lines) != codes.shape[0]:
            raise ValueError(f"Mismatch between number of lines ({len(lines)}) and codes ({codes.shape[0]}).")

        self.line_to_code: Dict[str, str] = {
            lines[i]: " ".join(map(str, codes[i])) for i in range(len(lines))
        }

        # Verification: Check for duplicate codes if all original lines were unique.
        # This assertion depends on the RVQ's ability to produce unique codes,
        # which is a goal but not always guaranteed if max_layers or stagnation limits are hit.
        if len(set(lines)) == len(lines) and len(self.line_to_code) != len(set(self.line_to_code.values())):
            logger.warning(
                f"Duplicate RVQ codes generated for unique input lines. "
                f"Unique lines: {len(self.line_to_code)}, Unique codes: {len(set(self.line_to_code.values()))}."
                " This might be expected if RVQ could not fully differentiate all inputs."
            )
        elif len(set(lines)) == len(lines):
            logger.info(f"All {len(self.line_to_code)} unique input lines mapped to unique RVQ codes.")

    def fit(self, lines):
        """
        Fit the RVQ model to the input data.
        This method generates embeddings for the input data and applies
        K-Means clustering iteratively to learn the quantization layers.
        The fitting process continues until either all input vectors have
        unique codes or a stagnation condition is met.
        The final output is a matrix of quantized codes, where each row
        corresponds to an input vector and each column corresponds to a
        quantization layer.
        Args:
            lines (List[str]): List of input strings (text, molecules, proteins).
        """
        X = self.generate_embeddings(lines)

        if X.ndim != 2:
            raise ValueError(f"Input array X must be 2-dimensional, got {X.ndim} dimensions.")

        self.quantizers_ = []
        self.n_layers_fitted_ = 0
        self.is_fitted_ = False

        layered_labels: List[np.ndarray] = []
        residual = X.copy()
        num_samples = X.shape[0]

        stagnation_counter = 0
        previous_n_unique_codes = 0

        for layer_idx in range(self.max_layers):
            self.n_layers_fitted_ += 1
            logger.info(f"Fitting RVQ Layer {self.n_layers_fitted_}...")

            current_residual_std = np.std(residual)
            if current_residual_std < 1e-9:  # very small residuals, further quantization might not be meaningful
                logger.warning(
                    f"Layer {self.n_layers_fitted_}: Residual standard deviation is very small ({current_residual_std:.2e}). "
                    "Further quantization might not be effective. Stopping."
                )
                self.n_layers_fitted_ -= 1  # Decrement as this layer won't be added
                break
            if self.random_fit:
                layer_predictions = np.random.randint(0, self.n_clusters, size=num_samples)
                centroids = np.zeros_like(residual)
            else:
                kmeans = KMeans(
                    n_clusters=self.n_clusters,
                    random_state=self.random_state,
                    n_init='auto',  # Suppress future warning for Kmeans
                    **self.kmeans_kwargs,
                )
                kmeans.fit(residual)
                self.quantizers_.append(kmeans)
                layer_predictions = kmeans.predict(residual)
                centroids = kmeans.cluster_centers_[layer_predictions]

            layered_labels.append(layer_predictions)
            residual = residual - centroids

            # Check for unique codes
            current_codes_as_tuples = self._convert_labels_to_comparable_codes(layered_labels)
            current_n_unique_codes = len(set(current_codes_as_tuples))

            logger.info(
                f"Layer {self.n_layers_fitted_}: {current_n_unique_codes}/{num_samples} unique codes found."
            )

            if current_n_unique_codes == num_samples:
                logger.info(
                    f"All {num_samples} samples have unique codes. Stopping RVQ fitting."
                )
                break

            if current_n_unique_codes == previous_n_unique_codes:
                stagnation_counter += 1
                logger.info(
                    f"Layer {self.n_layers_fitted_}: No improvement in unique codes. Stagnation step {stagnation_counter}."
                )
                if stagnation_counter >= self.max_stagnation_steps:
                    if self.noise_factor > 0 and current_residual_std > 1e-9:  # Add noise only if std is not too small
                        logger.warning(
                            f"Layer {self.n_layers_fitted_}: Max stagnation steps reached. Adding random noise to residuals."
                        )
                        noise_std = self.noise_factor * current_residual_std
                        residual += np.random.normal(0, noise_std, residual.shape)
                        stagnation_counter = 0  # Reset counter after adding noise
                    else:
                        logger.warning(
                            f"Layer {self.n_layers_fitted_}: Max stagnation steps reached and noise addition is disabled or residual std is too small. Stopping."
                        )
                        break
            else:
                stagnation_counter = 0
                previous_n_unique_codes = current_n_unique_codes

            if layer_idx == self.max_layers - 1:
                logger.info(f"Reached maximum allowed layers ({self.max_layers}). Stopping.")

        # Combine labels from all layers into the final codes matrix
        codes = np.zeros((num_samples, self.n_layers_fitted_), dtype=int)
        for i in range(self.n_layers_fitted_):
            codes[:, i] = layered_labels[i]

        self.is_fitted_ = True
        logger.info(f"RVQ fitting completed with {self.n_layers_fitted_} layers.")
        self.create_line_to_code_map(lines, codes)

    def transform(self, lines: List[str]) -> np.ndarray:
        """
        Transform input lines into their corresponding RVQ codes.

        Args:
            lines (List[str]): List of input strings (text, molecules, proteins).
        Returns:
            np.ndarray: A numpy array of RVQ codes.
        """
        if not self.is_fitted_:
            raise ValueError("RVQ model is not fitted. Please call fit() before transform().")

        for line in lines:
            if line not in self.line_to_code:
                raise ValueError(
                    f"Line '{line}' not found in fitted lines. Ensure the line was part of the training set.")

        codes = np.array([self.line_to_code[line] for line in lines])
        return codes

