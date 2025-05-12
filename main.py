import torch
from transformers import BertGenerationDecoder, BertGenerationConfig, BertGenerationEncoder
from transformers import Trainer, TrainingArguments
from prefix_trie import build_mask_from_trie, build_trie_from_text
import numpy as np
from torch.nn import CrossEntropyLoss
from pretrained_manager import get_model_and_tokenizer, QuantizeTokenizer, MOLECULES, PROTEINS, REACTIONS, MODEL_TO_DIM
from data_manager import get_dti_datasets, SrcTgtDataset
from rvq import ResidualVectorQuantizer
from utils import EvalLoggingCallback

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def get_bert_encoder(tokenizer, num_hidden_layers, num_attention_heads, dropout, encoder_dim):
    encoder_config = BertGenerationConfig(
        vocab_size=len(tokenizer.get_vocab()),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        is_encoder_decoder=True,
        is_decoder=False,
        add_cross_attention=False,
        hidden_size=encoder_dim,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=encoder_dim * 4,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        max_position_embeddings=256,
    )
    encoder = BertGenerationEncoder(encoder_config)
    return encoder


def get_encoder_decoder(src_model_type, src_model_name, tgt_model_type, tgt_model_name,
                        num_hidden_layers, num_attention_heads, hidden_size,
                        dropout, encoder_dim, train_encoder, pretrained_encoder):
    src_model, src_tokenizer = get_model_and_tokenizer(src_model_type, src_model_name)
    tgt_model, tgt_tokenizer = get_model_and_tokenizer(tgt_model_type, tgt_model_name)

    if not pretrained_encoder:
        src_model = get_bert_encoder(src_tokenizer, num_hidden_layers, num_attention_heads, dropout, encoder_dim)

    src_model.to(device)
    if train_encoder:
        src_model.train()
        for param in src_model.parameters():
            param.requires_grad = True
    else:
        src_model.eval()
        for param in src_model.parameters():
            param.requires_grad = False

    # Load the pretrained decoder
    decoder_config = BertGenerationConfig(
        vocab_size=len(tgt_tokenizer.get_vocab()),
        eos_token_id=tgt_tokenizer.eos_token_id,
        pad_token_id=tgt_tokenizer.pad_token_id,
        bos_token_id=tgt_tokenizer.bos_token_id,
        decoder_start_token_id=tgt_tokenizer.pad_token_id,
        is_encoder_decoder=True,
        is_decoder=True,
        add_cross_attention=True,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=hidden_size * 4,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        max_position_embeddings=512,
    )
    decoder = BertGenerationDecoder(decoder_config)
    decoder.train().to(device)

    return src_model, src_tokenizer, tgt_model, tgt_tokenizer, decoder


def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = predictions.argmax(-1)
    predictions = predictions[:, :-1]
    labels = labels[:, 1:]

    non_pad_mask = labels != -100
    token_correct = 0
    token_total = 0
    sample_correct = 0
    sample_total = len(labels)

    for i in range(len(labels)):
        # Get mask for this sequence
        seq_mask = non_pad_mask[i]

        # Extract non-padded tokens for this sequence
        seq_true = labels[i][seq_mask]
        seq_pred = predictions[i][seq_mask]

        # Count correct tokens
        token_correct += np.sum(seq_pred == seq_true)
        token_total += len(seq_true)

        # Check if entire sequence is correct (exact match)
        if np.array_equal(seq_pred, seq_true):
            sample_correct += 1

    # Calculate accuracies
    token_accuracy = token_correct / token_total if token_total > 0 else 0
    sample_accuracy = sample_correct / sample_total if sample_total > 0 else 0

    return {
        "token_accuracy": token_accuracy,
        "sample_accuracy": sample_accuracy
    }


def update_output_with_trie(decoder_outputs, input_ids, trie, vocab_size, labels=None, entropy_normalize=False):
    trie_mask = build_mask_from_trie(trie, input_ids, vocab_size)

    trie_mask = trie_mask[:, :-1, :]
    trie_mask_out = trie_mask.sum(dim=-1) <= 1
    decoder_outputs.trie_mask_out = trie_mask_out
    valid_token_count = trie_mask.sum(dim=-1)

    trie_mask = trie_mask.masked_fill(trie_mask == 0, -1e6)
    trie_mask = trie_mask.masked_fill(trie_mask == 1, 0)
    trie_mask = trie_mask.to(decoder_outputs.logits.device)
    decoder_outputs.logits[:, :-1] += trie_mask
    if labels is not None:
        labels[:, 1:][trie_mask_out] = -100
        if entropy_normalize:
            information_weights = torch.log(valid_token_count + 1 + 1e-6)  # add 2 to avoid log(1)=0
            info_weights_expanded = information_weights.unsqueeze(-1)
            info_weights_expanded = info_weights_expanded.to(decoder_outputs.logits.device)
            normalized_logits = decoder_outputs.logits[:, :-1] / info_weights_expanded
            decoder_outputs.logits[:, :-1] = normalized_logits

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        decoder_outputs.loss = loss_fct(
            decoder_outputs.logits[:, :-1].reshape(-1, decoder_outputs.logits[:, :-1].size(-1)),
            labels[:, 1:].reshape(-1))
    return decoder_outputs


class BaseEncoderDecoderModel(torch.nn.Module):
    """Base class containing common functionality for encoder-decoder models."""

    def __init__(self, decoder, trie, encoder_dim, bottleneck_dim,
                 entropy_normalize, constraint):
        super(BaseEncoderDecoderModel, self).__init__()
        self.decoder = decoder
        self.trie = trie
        self.entropy_normalize = entropy_normalize
        self.constraint = constraint

        # Create encoder projection layer(s)
        if bottleneck_dim > 0:
            self.encoder_project = torch.nn.Sequential(
                torch.nn.Linear(encoder_dim, bottleneck_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(bottleneck_dim, self.decoder.config.hidden_size)
            )
        else:
            self.encoder_project = torch.nn.Linear(
                encoder_dim, self.decoder.config.hidden_size
            )

    def project_encoder_outputs(self, encoder_hidden_states):
        """Project encoder outputs to decoder hidden dimension."""
        return self.encoder_project(encoder_hidden_states)

    def apply_trie_constraints(self, decoder_outputs, input_ids, labels=None):
        """Apply trie constraints to decoder outputs if needed."""
        if self.trie is None:
            return decoder_outputs

        apply_constraint = (self.constraint == 0 or
                            (self.constraint == 2 and not self.training))

        if apply_constraint:
            return update_output_with_trie(
                decoder_outputs,
                input_ids,
                self.trie,
                self.decoder.config.vocab_size,
                labels,
                entropy_normalize=self.entropy_normalize and self.training,
            )

        return decoder_outputs


class EndToEndModel(BaseEncoderDecoderModel):
    def __init__(self, encoder, decoder, trie, encoder_dim, bottleneck_dim, pooling,
                 entropy_normalize, constraint):
        super(EndToEndModel, self).__init__(
            decoder=decoder,
            trie=trie,
            encoder_dim=encoder_dim,
            bottleneck_dim=bottleneck_dim,
            entropy_normalize=entropy_normalize,
            constraint=constraint
        )
        self.encoder = encoder
        self.pooling = pooling

    def forward(self, src_input_ids, src_attention_mask, input_ids, attention_mask, labels=None):
        # Run through encoder
        encoder_outputs = self.encoder(input_ids=src_input_ids, attention_mask=src_attention_mask)

        # Handle pooling if needed
        if self.pooling:
            if hasattr(encoder_outputs, "pooler_output") and encoder_outputs.pooler_output is not None:
                encoder_hidden_states = encoder_outputs.pooler_output.unsqueeze(1)
            else:
                encoder_hidden_states = encoder_outputs.last_hidden_state.mean(dim=1)
            if encoder_hidden_states.ndim == 2:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
            encoder_attention_mask = torch.ones(src_attention_mask.shape[0], 1).to(src_attention_mask.device)
        else:
            encoder_hidden_states = encoder_outputs.last_hidden_state
            encoder_attention_mask = src_attention_mask

        # Project encoder outputs
        projected_encoder_hidden_states = self.project_encoder_outputs(encoder_hidden_states)

        # Run through decoder
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states=projected_encoder_hidden_states,
            labels=labels,
        )

        # Apply trie constraints if needed
        return self.apply_trie_constraints(decoder_outputs, input_ids, labels)


class EnzymeDecoder(BaseEncoderDecoderModel):
    def __init__(self, decoder, trie, encoder_dim, bottleneck_dim,
                 entropy_normalize, constraint):
        super(EnzymeDecoder, self).__init__(
            decoder=decoder,
            trie=trie,
            encoder_dim=encoder_dim,
            bottleneck_dim=bottleneck_dim,
            entropy_normalize=entropy_normalize,
            constraint=constraint
        )

    def forward(self, input_ids, attention_mask, encoder_outputs, encoder_attention_mask, labels):
        # Project encoder outputs
        projected_encoder_outputs = self.project_encoder_outputs(encoder_outputs)

        # Run through decoder
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states=projected_encoder_outputs,
            labels=labels,
        )

        # Apply trie constraints if needed
        return self.apply_trie_constraints(decoder_outputs, input_ids, labels)


if __name__ == "__main__":
    from utils import get_config

    config = get_config()

    if config['dataset'] == "care":
        from data_manager import get_care_datasets

        train_pos_src, train_pos_tgt, test_pos_src, test_pos_tgt = get_care_datasets()
        valid_pos_src, valid_pos_tgt = [], []
        train_neg_src, train_neg_tgt, valid_neg_src, valid_neg_tgt, test_neg_src, test_neg_tgt = [], [], [], [], [], []
    elif config['dataset'] == "ddi":
        from data_manager import get_ddi_datasets

        train_pos_src, train_pos_tgt, train_neg_src, train_neg_tgt, valid_pos_src, valid_pos_tgt, valid_neg_src, valid_neg_tgt, test_pos_src, test_pos_tgt, test_neg_src, test_neg_tgt = get_ddi_datasets()
    else:
        train_pos_src, train_pos_tgt, train_neg_src, train_neg_tgt, valid_pos_src, valid_pos_tgt, valid_neg_src, valid_neg_tgt, test_pos_src, test_pos_tgt, test_neg_src, test_neg_tgt = get_dti_datasets(
            config["dataset"], cold_fasta=config["cold_fasta"], cold_smiles=config["cold_smiles"])
    if config["replace_src_target"]:
        train_pos_src, train_pos_tgt = train_pos_tgt, train_pos_src
        train_neg_src, train_neg_tgt = train_neg_tgt, train_neg_src
        valid_pos_src, valid_pos_tgt = valid_pos_tgt, valid_pos_src
        valid_neg_src, valid_neg_tgt = valid_neg_tgt, valid_neg_src
        test_pos_src, test_pos_tgt = test_pos_tgt, test_pos_src
        test_neg_src, test_neg_tgt = test_neg_tgt, test_neg_src

    src_model, src_tokenizer, tgt_model, tgt_tokenizer, decoder = get_encoder_decoder(
        config["src_model_type"], config["src_model_name"],
        config["tgt_model_type"], config["tgt_model_name"],
        config["num_hidden_layers"],
        config["num_attention_heads"],
        config["hidden_size"],
        config["dropout"], config["encoder_dim"],
        config["train_encoder"],
        config["pretrained_encoder"])
    all_tgts = list(set(train_pos_tgt + train_neg_tgt + valid_pos_tgt + valid_neg_tgt + test_pos_tgt + test_neg_tgt))

    if config["quantize"]:
        RVQ = ResidualVectorQuantizer(n_clusters=config["n_clusters"], model=tgt_model, tokenizer=tgt_tokenizer,
                                      random_fit=config["random_tgt"])

        RVQ.fit(all_tgts)
        train_pos_tgt = RVQ.transform(train_pos_tgt)
        valid_pos_tgt = RVQ.transform(valid_pos_tgt)
        test_pos_tgt = RVQ.transform(test_pos_tgt)
        train_neg_tgt = RVQ.transform(train_neg_tgt)
        valid_neg_tgt = RVQ.transform(valid_neg_tgt)
        test_neg_tgt = RVQ.transform(test_neg_tgt)
        all_tgts = RVQ.transform(all_tgts)
        tgt_tokenizer = QuantizeTokenizer(max_token=config["n_clusters"])

        decoder.resize_token_embeddings(len(tgt_tokenizer.get_vocab()))

    encoder_dim = MODEL_TO_DIM.get(config["src_model_name"], 512)
    train_dataset = SrcTgtDataset(train_pos_src, train_pos_tgt, src_tokenizer, tgt_tokenizer, src_encoder=src_model,
                                  max_length=256, pooling=config['pooling'], train_encoder=config["train_encoder"])
    valid_dataset = SrcTgtDataset(valid_pos_src, valid_pos_tgt, src_tokenizer, tgt_tokenizer, max_length=256,
                                  src_encoder=src_model, pooling=config['pooling'], train_encoder=config["train_encoder"])
    test_dataset = SrcTgtDataset(test_pos_src, test_pos_tgt, src_tokenizer, tgt_tokenizer, max_length=256,
                                 src_encoder=src_model, pooling=config['pooling'], train_encoder=config["train_encoder"])

    trie = build_trie_from_text(list(set(all_tgts)), tgt_tokenizer)

    # Choose the appropriate model based on whether we're training the encoder
    common_model_args = dict(decoder=decoder,
                             trie=trie,
                             encoder_dim=encoder_dim,
                             entropy_normalize=config["entropy_normalize"],
                             constraint=config["constraint"],
                             bottleneck_dim=config["buttleneck_dim"],
                             )

    if config["train_encoder"]:
        model = EndToEndModel(
            encoder=src_model,
            **common_model_args
        )
    else:
        model = EnzymeDecoder(
            **common_model_args
        )

    if config["dataset"] != "care":
        from logit_feature_transformer_pipeline import evaluate_model, evaluate_model_logits

        neg_valid_dataset = SrcTgtDataset(valid_neg_src, valid_neg_tgt, src_tokenizer, tgt_tokenizer, max_length=256,
                                          src_encoder=src_model, pooling=config['pooling'], train_encoder=config["train_encoder"])
        neg_test_dataset = SrcTgtDataset(test_neg_src, test_neg_tgt, src_tokenizer, tgt_tokenizer, max_length=256,
                                         src_encoder=src_model, pooling=config['pooling'], train_encoder=config["train_encoder"])

        if config['dataset'] == "ddi":
            compute_metrics_func = lambda x: evaluate_model_logits(test_pos_dataset=test_dataset,
                                                                   test_neg_dataset=neg_test_dataset, model=model,
                                                                   batch_size=config["batch_size"])
        else:

            compute_metrics_func = lambda x: evaluate_model(valid_pos_dataset=valid_dataset,
                                                            valid_neg_dataset=neg_valid_dataset,
                                                            test_pos_dataset=test_dataset,
                                                            test_neg_dataset=neg_test_dataset, model=model,
                                                            batch_size=config["batch_size"],
                                                            d_model=config["meta_d_model"],
                                                            nhead=config["meta_nhead"],
                                                            num_layers=config["meta_num_layers"],
                                                            dropout=config["meta_dropout"],
                                                            learning_rate=config["meta_learning_rate"],
                                                            weight_decay=config["meta_weight_decay"],
                                                            num_epochs=config["meta_num_epochs"],
                                                            patience=config["meta_patience"])

        test_dataset_dummy = torch.utils.data.Subset(test_dataset, [0])
        eval_dataset = {"valid": test_dataset_dummy}
        metric_for_best_model = "eval_valid_auc"

    else:

        compute_metrics_func = lambda x: compute_metrics(x)
        train_small_indices = np.random.choice(len(train_dataset), len(test_dataset), replace=False)
        train_small_dataset = torch.utils.data.Subset(train_dataset, train_small_indices)
        eval_dataset = {"test": test_dataset, "train": train_small_dataset}
        metric_for_best_model = "eval_test_token_accuracy"

    print("Src model:")
    print(src_model)
    print("Number of parameters:", sum(p.numel() for p in src_model.parameters() if p.requires_grad))
    print("model:")
    print(model)
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    output_dir = config["output_dir"]
    logs_dir = output_dir.replace("results", "logs")
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logs_dir,
        evaluation_strategy="steps",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        eval_accumulation_steps=30,
        save_total_limit=3,
        max_steps=config["steps"],
        logging_steps=config["log_steps"],
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        lr_scheduler_type="constant",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        report_to=[config["report_to"]],
        save_safetensors=False,
        auto_find_batch_size=True,
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_func
    )
    eval_logging_callback = EvalLoggingCallback(output_dir=output_dir)
    trainer.add_callback(eval_logging_callback)
    # trainer.evaluate()
    # Train model
    print("Training model...")
    trainer.train()
    print("Training complete!")
