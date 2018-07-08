from __future__ import absolute_import, division, print_function

import tensorflow as tf

def create_standard_hparams():
    return tf.contrib.training.HParams(
        # Data
        src="",
        tgt="",
        train_prefix="",
        dev_prefix="",
        test_prefix="",
        embed_prefix="",
        vocab_prefix="",
        out_dir="",

        # Networks
        num_units=512,
        num_layers=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.2,
        unit_type="lstm",
        encoder_type="bi",
        residual=False,
        time_major=True,
        num_embeddings_partition=0,

        # Attention Mechanism
        attention="scaled_luong",
        attention_architecture="standard",
        output_attention=True,
        pass_hidden_state=True,

        # Train
        optimizer="sgd",
        batch_size=128,
        init_op="uniform",
        init_weight=0.1,
        max_gradient_norm=5.0,
        learning_rate=1.0,
        warmup_steps=0,
        warmup_scheme="t2t",
        decay_scheme="luong234",
        colocate_gradients_with_ops=True,
        num_train_steps=12000,

        # Data Constraints
        num_buckets=5,
        max_train=0,
        src_max_len=50,
        tgt_max_len=50,
        src_max_len_infer=0,
        tgt_max_len_infer=0,

        # Data Format
        sos="<s>",
        eos="</s>",
        subword_option="",
        check_special_token=True,

        # Misc
        forget_bias=1.0,
        num_gpus=0,
        epoch_step=0,
        steps_per_stats=100,
        steps_per_external_eval=0,
        share_vocab=False,
        metrics=["bleu"],
        log_device_placement=False,
        random_seed=None,
        # Only enable beam search during inference when beam_width > 0
        beam_width=0,
        length_penalty_weight=0.0,
        override_loaded_hparams=True,
        num_keep_ckpts=5,
        avg_ckpts=False,

        # For Inference
        inference_indices=None,
        infer_batch_size=32,
        sampling_temperature=0.0,
        num_translations_per_input=1,
    )
