from rxnfp.main import get_model_and_tokenizer as get_model_and_tokenizer_rxnfp
from transformers import AutoTokenizer, AutoModel
import torch

MOLECULES = "molecules"
REACTIONS = "reactions"
PROTEINS = "proteins"



MODEL_TO_DIM={
    "rxnfp": 256,
    "ibm/MoLFormer-XL-both-10pct": 768,
    "facebook/esm2_t33_650M_UR50D": 1280,
}

def get_model_and_tokenizer(data_type: str, model_name, quantize=False):
    """
    Retrieves the model and tokenizer based on the type of input (molecules or text).

    Args:
        data_type (str): Type of data ("molecules" or "text").
        model_name (str): Name of the model to be used.
    Returns:
        tokenizer: The tokenizer for the specified model.
        model: The model for the specified type of data.
    """
    if data_type == REACTIONS:
        model, tokenizer = get_model_and_tokenizer_rxnfp()
    else:
        args = {}
        if "MoLFormer" in model_name:
            args["trust_remote_code"] = True
            args["deterministic_eval"] = True

        model = AutoModel.from_pretrained(model_name, **args)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **args)
    if quantize:
        tokenizer = QuantizeTokenizer()
    return model, tokenizer


class QuantizeTokenizer:
    def __init__(self, max_token=15):
        self.eos_token_id = max_token
        self.pad_token_id = max_token + 1
        self.bos_token_id = max_token + 2
        self.vocab_size = max_token + 3

    def __len__(self):
        return self.vocab_size

    def get_vocab(self):
        return {i: i for i in range(self.vocab_size)}

    def __call__(self, seq, **kwargs):
        seq = torch.LongTensor([self.bos_token_id] + [int(x) for x in seq.split()] + [self.pad_token_id]).unsqueeze(0)
        mask = torch.ones(seq.shape)
        return {"input_ids": seq, "attention_mask": mask}

    def encode(self, seq, **kwargs):
        return self(seq, **kwargs)["input_ids"][0].tolist()

    def decode(self, seq):
        return " ".join([str(x) for x in seq])
