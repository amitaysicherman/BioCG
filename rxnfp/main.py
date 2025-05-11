from rxnfp.tokenizer import SmilesTokenizer
from transformers import BertModel
import torch

def get_model_and_tokenizer():
    reaction_tokenizer = SmilesTokenizer.from_pretrained("rxnfp/data/vocab.txt")
    reaction_model = BertModel.from_pretrained("rxnfp/data")
    return reaction_model,reaction_tokenizer


if __name__ == "__main__":
    model, tokenizer = get_model_and_tokenizer()
    print("Model:", model)
    print("Number of parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
    print("Tokenizer:", tokenizer)

    example_reaction = "CC(=O)O>>CC(=O)N"
    tokens = tokenizer(example_reaction, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True, output_attentions=True)
    print("Model Outputs:", outputs.keys())
    print("Model Outputs last_hidden_state:", outputs.last_hidden_state.shape)
    print("Model Outputs pooler_output:", outputs.pooler_output.shape)
    print("Model Outputs hidden_states:", outputs.hidden_states[-1].shape)
    print("Model Outputs attentions:", outputs.attentions[-1].shape)
