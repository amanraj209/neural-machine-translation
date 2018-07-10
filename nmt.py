from __future__ import print_function

import os
import sys
import random
import argparse
import numpy as np
import tensorflow as tf

import inference, train
from utils import evaluation_utils, vocab_utils, misc_utils as utils

utils.check_tensorflow_version()


def create_hparams():
    return tf.contrib.training.HParams(
        src="vi",
        tgt="en",
        train_prefix="./data/train",
        dev_prefix="./data/tst2012",
        test_prefix="./data/tst2013",
        vocab_prefix="./data/vocab",
        embed_prefix=None,
        out_dir="./nmt_model",

        num_units=128,
        num_layers=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.2,
        unit_type="lstm",
        encoder_type="bi",
        residual=False,
        time_major=True,
        num_embeddings_partitions=0,

        attention="scaled_luong",
        attention_architecture="standard",
        output_attention=True,
        pass_hidden_state=True,

        optimizer="sgd",
        num_train_steps=12000,
        batch_size=128,
        init_op="uniform",
        init_weight=0.1,
        max_gradient_norm=5.0,
        learning_rate=0.1,
        warmup_steps=0,
        warmup_scheme="t2t",
        decay_scheme="luong234",
        colocate_gradients_with_ops=True,

        num_buckets=5,
        max_train=0,
        src_max_len=50,
        tgt_max_len=50,

        src_max_len_infer=0,
        tgt_max_len_infer=0,
        infer_batch_size=32,
        beam_width=3,
        length_penalty_weight=0.0,
        sampling_temperature=0.0,
        num_translations_per_input=1,

        sos="<s>",
        eos="</s>",
        subword_option="",
        check_special_token=True,

        forget_bias=1.0,
        num_gpus=2,
        epoch_step=0,
        steps_per_stats=100,
        steps_per_external_eval=0,
        share_vocab=False,
        metrics=["bleu"],
        log_device_placement=False,
        random_seed=None,
        override_loaded_hparams=True,
        num_keep_ckpts=5,
        avg_ckpts=True,
        num_intra_threads=0,
        num_inter_threads=0,

        inference_input_file=None,
        inference_indices=None,
        inference_list=None,
        inference_output_file=None,
        ckpt=None,
        inference_ref_file=None,
        hparams_path=None
    )


def extend_hparams(hparams):
    assert hparams.num_encoder_layers and hparams.num_decoder_layers
    if hparams.num_encoder_layers != hparams.num_decoder_layers:
        hparams.pass_hidden_state = False
        utils.print_out("Num encoder layer %d is different from num decoder layer %d, so set pass_hidden_state to False" % (
                        hparams.num_encoder_layers, hparams.num_decoder_layers))
    
    if hparams.encoder_type == "bi" and hparams.num_encoder_layers % 2 != 0:
        raise ValueError("For bi, num_encoder_layers %d should be even" % hparams.num_encoder_layers)

    num_encoder_residual_layers = 0
    num_decoder_residual_layers = 0
    if hparams.residual:
        if hparams.num_encoder_layers > 1:
            num_encoder_residual_layers = hparams.num_encoder_layers - 1
        if hparams.num_decoder_layers > 1:
            num_decoder_residual_layers = hparams.num_decoder_layers - 1
    
    hparams.add_hparam("num_encoder_residual_layers", num_encoder_residual_layers)
    hparams.add_hparam("num_decoder_residual_layers", num_decoder_residual_layers)

    if hparams.subword_option and hparams.subword_option not in ["spm", "bpe"]:
        raise ValueError("subword option must be either spm, or bpe")
    
    utils.print_out("# hparams:")
    utils.print_out("src=%s" % hparams.src)
    utils.print_out("tgt=%s" % hparams.tgt)
    utils.print_out("train_prefix=%s" % hparams.train_prefix)
    utils.print_out("dev_prefix=%s" % hparams.dev_prefix)
    utils.print_out("test_prefix=%s" % hparams.test_prefix)
    utils.print_out("out_dir=%s" % hparams.out_dir)

    if hparams.vocab_prefix:
        src_vocab_file = hparams.vocab_prefix + "." + hparams.src
        tgt_vocab_file = hparams.vocab_prefix + "." + hparams.tgt
    else:
        raise ValueError("hparams.vocab_prefix must be provided.")

    src_vocab_size, src_vocab_file = vocab_utils.check_vocab(src_vocab_file, hparams.out_dir,
                                                             check_special_token=hparams.check_special_token,
                                                             sos=hparams.sos, eos=hparams.eos, unk=vocab_utils.UNK)

    if hparams.share_vocab:
        utils.print_out("using source vocab for target")
        tgt_vocab_file = src_vocab_file
        tgt_vocab_size = src_vocab_size
    else:
        tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(tgt_vocab_file, hparams.out_dir,
                                                                 check_special_token=hparams.check_special_token,
                                                                 sos=hparams.sos, eos=hparams.eos, unk=vocab_utils.UNK)
    hparams.add_hparam("src_vocab_size", src_vocab_size)
    hparams.add_hparam("tgt_vocab_size", tgt_vocab_size)
    hparams.add_hparam("src_vocab_file", src_vocab_file)
    hparams.add_hparam("tgt_vocab_file", tgt_vocab_file)

    hparams.add_hparam("src_embed_file", "")
    hparams.add_hparam("tgt_embed_file", "")
    if hparams.embed_prefix:
        src_embed_file = hparams.embed_prefix + "." + hparams.src
        tgt_embed_file = hparams.embed_prefix + "." + hparams.tgt

        if tf.gfile.Exists(src_embed_file):
            hparams.src_embed_file = src_embed_file

        if tf.gfile.Exists(tgt_embed_file):
            hparams.tgt_embed_file = tgt_embed_file

    if not tf.gfile.Exists(hparams.out_dir):
        utils.print_out("# Creating output directory %s ..." % hparams.out_dir)
        tf.gfile.MakeDirs(hparams.out_dir)

    for metric in hparams.metrics:
        hparams.add_hparam("best_" + metric, 0)
        best_metric_dir = os.path.join(hparams.out_dir, "best_" + metric)
        hparams.add_hparam("best_" + metric + "_dir", best_metric_dir)
        tf.gfile.MakeDirs(best_metric_dir)

        if hparams.avg_ckpts:
            hparams.add_hparam("avg_best_" + metric, 0) 
            best_metric_dir = os.path.join(hparams.out_dir, "avg_best_" + metric)
            hparams.add_hparam("avg_best_" + metric + "_dir", best_metric_dir)
            tf.gfile.MakeDirs(best_metric_dir)

    return hparams


def ensure_compatible_hparams(hparams, default_hparams, hparams_path=None):
    default_hparams = utils.maybe_parse_standard_hparams(default_hparams, hparams_path)

    default_config = default_hparams.values()
    config = hparams.values()
    for key in default_config:
        if key not in config:
            hparams.add_hparam(key, default_config[key])

    if default_hparams.override_loaded_hparams:
        for key in default_config:
            if getattr(hparams, key) != default_config[key]:
                utils.print_out("# Updating hparams.%s: %s -> %s" %
                                (key, str(getattr(hparams, key)),
                                str(default_config[key])))
                setattr(hparams, key, default_config[key])
    return hparams


def create_or_load_hparams(out_dir, default_hparams, hparams_path, save_hparams=True):
    hparams = utils.load_hparams(out_dir)
    if not hparams:
        hparams = default_hparams
        hparams = utils.maybe_parse_standard_hparams(hparams, hparams_path)
        hparams = extend_hparams(hparams)
    else:
        hparams = ensure_compatible_hparams(hparams, default_hparams, hparams_path)

    if save_hparams:
        utils.save_hparams(out_dir, hparams)
        for metric in hparams.metrics:
            utils.save_hparams(getattr(hparams, "best_" + metric + "_dir"), hparams)

    utils.print_hparams(hparams)
    return hparams


def run_main(default_hparams, train_fn, inference_fn, target_session=""):
    job_id = 0
    num_workers = 1
    utils.print_out("job id %d" % job_id)

    random_seed = None
    if random_seed is not None and random_seed > 0:
        utils.print_out("set random seed to %d" % random_seed)
        random.seed(random_seed + job_id)
        np.random.seed(random_seed + job_id)

    out_dir = "./nmt_model"
    if not tf.gfile.Exists(out_dir): tf.gfile.MakeDirs(out_dir)

    hparams = create_or_load_hparams(out_dir, default_hparams, default_hparams.hparams_path, save_hparams=(job_id == 0))

    if hparams.inference_input_file:
        hparams.inference_indices = None
        if hparams.inference_list:
            hparams.inference_indices = [int(token)  for token in hparams.inference_list.split(",")]

        trans_file = hparams.inference_output_file
        ckpt = hparams.ckpt
        if not ckpt:
            ckpt = tf.train.latest_checkpoint(out_dir)
            inference_fn(ckpt, hparams.inference_input_file, trans_file, hparams, num_workers, job_id)

        ref_file = hparams.inference_ref_file
        if ref_file and tf.gfile.Exists(trans_file):
            for metric in hparams.metrics:
                score = evaluation_utils.evaluate(ref_file, trans_file, metric, hparams.subword_option)
                utils.print_out("%s: %.1f" % (metric, score))
    else:
        train_fn(hparams, target_session=target_session)


def main(unused_argv):
    default_hparams = create_hparams()
    train_fn = train.train
    inference_fn = inference.inference
    run_main(default_hparams, train_fn, inference_fn)


if __name__ == "__main__":
    tf.app.run(main=main)
