# BioCG: Constrained Generative Modeling for Biochemical Interaction Prediction

This repository provides the official PyTorch implementation for the paper: **"BioCG: Constrained Generative Modeling for Biochemical Interaction Prediction"**

##  Abstract
![BioCG Framework](img/BioCG.png)

Predicting interactions between biochemical entities is a core challenge in drug discovery and systems biology, often hindered by limited data and poor generalization to unseen entities. We propose **BioCG (Biochemical Constrained Generation)**, a novel framework that reformulates interaction prediction as a constrained sequence generation task. BioCG encodes target entities as unique discrete sequences via Iterative Residual Vector Quantization (I-RVQ) and trains a generative model to produce the sequence of an interacting partner given a query entity. A trie-guided constrained decoding mechanism ensures all outputs are biochemically valid. An information-weighted training objective further focuses learning. BioCG achieves state-of-the-art (SOTA) performance across diverse tasks (DTI, DDI, Enzyme-Reaction Prediction), especially in cold-start conditions, offering a robust and data-efficient solution for in-silico biochemical discovery.

##  Key Features

* **Novel Generative Formulation**: Transforms interaction prediction into a constrained sequence generation problem over a finite catalog of valid biochemical entities.
* **Guaranteed Biochemical Validity**: Employs Iterative Residual Vector Quantization (I-RVQ) for unique target codes and a trie-guided decoder to ensure generated entities are always valid and known.
* **Superior Cold-Start Performance**: Excels in scenarios with unseen proteins or drugs, significantly outperforming previous SOTA. (e.g., **+14.3% AUC** on BioSNAP DTI for unseen proteins).
* **Data Efficiency**: The constrained approach and information-weighted loss enable robust learning even with limited paired experimental data.
* **Versatility**: Demonstrated SOTA results across multiple interaction types:
    * Drug-Target Interactions (DTI)
    * Drug-Drug Interactions (DDI)
    * Enzyme-Reaction Prediction

## ️ How BioCG Works

BioCG's methodology can be broken down into the following key stages:

1.  **Query Entity Encoding**:
    * Input query entities (drugs as SMILES, proteins as FASTA, reactions as SMILES) are encoded into fixed-size context vectors using pre-trained, domain-specific encoders (e.g., ESM for proteins, MolFormer for molecules, RXNFP for reactions). These encoders are typically frozen.

2.  **Target Entity Discretization (I-RVQ)**:
    * A finite catalog of target entities (e.g., all proteins in DrugBank, all EC numbers) is first embedded using appropriate pre-trained models.
    * Our **Iterative Residual Vector Quantization (I-RVQ)** module then generates unique discrete code sequences for each target. I-RVQ applies k-means clustering iteratively to residual vectors, adding layers until every target entity has a distinct code sequence. This creates a bijective mapping.

3.  **Trie Construction**:
    * A prefix tree (Trie) is built from all unique I-RVQ code sequences of the target entities. This trie represents the entire valid search space for the generative decoder.

4.  **Constrained Autoregressive Generation**:
    * A Transformer-based decoder autoregressively generates the target's I-RVQ code sequence, conditioned on the query entity's context vector.
    * At each generation step, the **Trie guides the decoding process**. A mask derived from the trie is applied to the decoder's output logits, restricting predictions to only valid next tokens that form a known target sequence.
    * The model is trained using an **Information-Weighted Loss** function, which prioritizes learning at more complex decision points (i.e., steps with higher branching factors in the trie).

5.  **Interaction Scoring**:
    * For evaluation, the likelihood of generating a specific target $t$ given a query $e$ is determined.
    * This can be a direct **Log Probability Score** from the generator.
    * Alternatively, a lightweight **Meta-Model** (Transformer-based) can be trained to take features from the generation process (token probabilities, branching factors, entropy) and output a calibrated binary interaction score.

##  Results Highlights

BioCG demonstrates significant improvements over existing methods across multiple benchmarks:

### Drug-Target Interaction (DTI) - BioSNAP Dataset

| Scenario         | BioCG (AUC)     | $\Delta$ vs. SOTA Baseline |
| :--------------- | :-------------- | :------------------------- |
| Seen             | **96.61%** | +2.70% (vs. Top-DTI)       |
| Unseen Protein   | **89.31%** | +14.30% (vs. Top-DTI)      |
| Unseen Drug      | 91.21%          | +2.15% (vs. DrugLAMP)      |
*(Top-DTI achieves 91.25% for Unseen Drug)*

### Drug-Drug Interaction (DDI) - DrugBank Dataset

| Metric | BioCG       | $\Delta$ vs. SOTA Baseline |
| :----- | :---------- | :------------------------- |
| ACC    | **77.23%** | +3.18% (vs. DDI-GCN)       |
| AUC    | **84.12%** | +3.21% (vs. MR-GNN)        |
| AUPR   | **72.16%** | +2.69% (vs. MR-GNN)        |

### Enzyme-Reaction Prediction - CARE Benchmark

| Metric   | BioCG       | $\Delta$ vs. SOTA Baseline |
| :------- | :---------- | :------------------------- |
| Top@1 Acc| **67.93%** | +7.61% (vs. CREEP)         |
| Top@2 Acc| **82.32%** | +2.86% (vs. CREEP)         |
| Top@3 Acc| **89.21%** | +1.63% (vs. CREEP)         |

## Datasets

BioCG was evaluated on the following publicly available datasets:

* **Drug-Target Interaction (DTI)**:
    * **BioSNAP Dataset**: Derived from DrugBank. Used for random splits and cold-start (unseen protein/drug) scenarios.
* **Drug-Drug Interaction (DDI)**:
    * **DrugBank Dataset**: Used for DDI prediction, including cold-start evaluation.
* **Enzyme-Reaction Prediction**:
    * **CARE Benchmark**: Reactions (SMILES) as queries, target enzymes identified by EC numbers (represented by protein sequences).

Dataset loading, preprocessing, and splitting are handled by `src/data_manager.py`. Please refer to the paper and configuration files for details on obtaining and structuring the data.

##  Getting Started

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ANON/BioCG.git](https://github.com/ANON/BioCG.git) 
    cd BioCG
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv biocg_env
    source biocg_env/bin/activate  # On Windows: biocg_env\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install PyTorch, Hugging Face Transformers, scikit-learn, fair-esm, and other necessary packages. Pre-trained models (ESM, MolFormer) are typically downloaded automatically on first use.


### Training

To train a BioCG model, run the main script with the path to your desired configuration file:

```bash
python src/main.py --config_path configs/your_experiment_config.yaml
