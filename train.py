from __future__ import print_function

import os
import time
import math
import random
import tensorflow as tf

from . import attention_model, model_util, inference, model as nmt_model
from .utils import nmt_utils, misc_utils as utils

utils.check_tensorflow_version()

__all__ = ["run_sample_decode", "run_internal_eval", "run_external_eval", "run_avg_external_eval",
           "run_full_eval", "init_stats", "update_stats", "print_step_info", "process_stats", "train"]


def _sample_decode(model, global_step, sess, hparams, iterator, src_data, tgt_data, iterator_src_placeholder,
                   iterator_batch_size_placeholder, summary_writer):
    decode_id = random.randint(0, len(src_data) - 1)
    utils.print_out("decoding %d" % decode_id)

    iterator_feed_dict = {
        iterator_src_placeholder: [src_data[decode_id]],
        iterator_batch_size_placeholder: 1
    }
    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
    nmt_outputs, attention_summary = model.decode(sess)

    if hparams.beam_width > 0:
        nmt_outputs = nmt_outputs[0]
    
    translation = nmt_utils.get_translation(nmt_outputs, sent_id=0, tgt_eos=hparams.eos, subword_option=hparams.subword_option)
    utils.print_out("src: %s" % src_data[decode_id])
    utils.print_out("ref: %s" % tgt_data[decode_id])
    utils.print_out(b"nmt: " + translation)

    if attention_summary is not None:
        summary_writer.add_summary(attention_summary, global_step)


def run_sample_decode(infer_model, infer_sess, model_dir, hparams, summary_writer, src_data, tgt_data):
    with infer_model.graph.as_default():
        loaded_infer_model, global_step = model_util.create_or_load_model(infer_model.model, model_dir,
                                                                          infer_sess, "infer")
    _sample_decode(loaded_infer_model, global_step, infer_sess, hparams, infer_model.iterator, src_data,
                    tgt_data, infer_model.src_placeholder, infer_model.batch_size_placeholder, summary_writer)


def _internal_eval(model, global_step, sess, iterator, iterator_feed_dict, summary_writer, label):
    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
    perplexity = model_util.compute_perplexity(model, sess, label)
    utils.add_summary(summary_writer, global_step, "%s_perplexity" % label, perplexity)
    return perplexity


# Compute internal evaluation (perplexity) for both dev/test
def run_internal_eval(eval_model, eval_sess, model_dir, hparams, summary_writer, use_test_set=True):
    with eval_model.graph.as_default():
        loaded_eval_model, global_step = model_util.create_or_load_model(eval_model.model, model_dir,
                                                                          eval_sess, "eval")
    dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
    dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
    dev_eval_iterator_feed_dict = {
        eval_model.src_file_placeholder: dev_src_file,
        eval_model.tgt_file_placeholder: dev_tgt_file
    }                

    dev_perplexity = _internal_eval(loaded_eval_model, global_step, eval_sess, eval_model.iterator,
                                    dev_eval_iterator_feed_dict, summary_writer, "dev")
    
    test_perplexity = None
    if use_test_set and hparams.test_prefix:
        test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
        test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
        test_eval_iterator_feed_dict = {
            eval_model.src_file_placeholder: test_src_file,
            eval_model.tgt_file_placeholder: test_tgt_file
        }                

        test_perplexity = _internal_eval(loaded_eval_model, global_step, eval_sess, eval_model.iterator,
                                        test_eval_iterator_feed_dict, summary_writer, "test")
    
    return dev_perplexity, test_perplexity


def _external_eval(model, global_step, sess, hparams, iterator, iterator_feed_dict, tgt_file,
                   label, summary_writer, save_on_best, avg_ckpts=False):
    out_dir = hparams.out_dir
    decode = global_step > 0
    if avg_ckpts:
        label = "avg_" + label
    if decode:
        utils.print_out("external evaluation, global step %d" % global_step)
    
    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

    output = os.path.join(out_dir, "output_%s" % label)
    scores = nmt_utils.decode_and_evaluate(label, model, sess, output, ref_file=tgt_file,
                                           metrics=hparams.metrics, subword_option=hparams.subword_option,
                                           beam_width=hparams.beam_width, tgt_eos=hparams.eos, decode=decode)
    
    if decode:
        for metric in hparams.metrics:
            if avg_ckpts:
                best_metric_label = "avg_best_" + metric
            else:
                best_metric_label = "best_" + metric
            
            utils.add_summary(summary_writer, global_step, "%s_%s" % (label, metric), scores[metric])

            if save_on_best and scores[metric] > getattr(hparams, best_metric_label):
                setattr(hparams, best_metric_label, scores[metric])
                model.saver.save(sess, os.path.join(getattr(hparams, best_metric_label + "_dir"), "translate.ckpt"),
                                 global_step=model.global_step)
        utils.save_hparams(out_dir, hparams)
    
    return scores


# Compute external evaluation like BLEU, ROUGE for both dev/test
def run_external_eval(infer_model, infer_sess, model_dir, hparams, summary_writer,
                      save_best_dev=True, use_test_set=True, avg_ckpts=False):
    with infer_model.graph.as_default():
        loaded_infer_model, global_step = model_util.create_or_load_model(infer_model.model, model_dir,
                                                                          infer_sess, "infer")
    dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
    dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
    dev_infer_iterator_feed_dict = {
        infer_model.src_placeholder: inference.load_data(dev_src_file),
        infer_model.batch_size_placeholder: hparams.infer_batch_size
    }     
    dev_scores = _external_eval(loaded_infer_model, global_step, infer_sess, hparams,
                                infer_model.iterator, dev_infer_iterator_feed_dict,
                                dev_tgt_file, "dev", summary_writer,
                                save_on_best=save_best_dev, avg_ckpts=avg_ckpts)
    
    test_scores = None
    if use_test_set and hparams.test_prefix:
        test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
        test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
        test_infer_iterator_feed_dict = {
            infer_model.src_placeholder: inference.load_data(test_src_file),
            infer_model.batch_size_placeholder: hparams.infer_batch_size
        }     
        test_scores = _external_eval(loaded_infer_model, global_step, infer_sess, hparams,
                                        infer_model.iterator, test_infer_iterator_feed_dict,
                                        test_tgt_file, "dev", summary_writer,
                                        save_on_best=save_best_dev, avg_ckpts=avg_ckpts)
    
    return dev_scores, test_scores, global_step


# Creates an averaged checkpoint and run external eval with it
def run_avg_external_eval(infer_model, infer_sess, model_dir, hparams, summary_writer, global_step):
    avg_dev_scores, avg_test_scores = None, None
    if hparams.avg_ckpts:
        global_step_name = infer_model.model.global_step_name.split(":")[0]
        avg_model_dir = model_util.avg_checkpoints(model_dir, hparams.num_keep_ckpts,
                                                   global_step, global_step_name)
        if avg_model_dir:
            avg_dev_scores, avg_test_scores, _ = run_external_eval(infer_model, infer_sess, avg_model_dir,
                                                                   hparams, summary_writer, avg_ckpts=True)
    return avg_dev_scores, avg_test_scores


def _format_results(name, perplexity, scores, metrics):
    result = ""
    if perplexity:
        result = "%s perplexity: %.2f" % (name, perplexity)
    if scores:
        for metric in metrics:
            if result:
                result += ", %s %s: %.1f" % (name, metric, scores[metric])
            else:
                result = ", %s %s: %.1f" % (name, metric, scores[metric])
    return result


# Wrapper for running sample_decode, internal_eval and external_eval
def run_full_eval(model_dir, infer_model, infer_sess, eval_model, eval_sess, hparams,
                  summary_writer, sample_src_data, sample_tgt_data, avg_ckpts=False):
    run_sample_decode(infer_model, infer_sess, model_dir, hparams, summary_writer,
                      sample_src_data, sample_tgt_data)
    
    dev_perplexity, test_perplexity = run_internal_eval(eval_model, eval_sess, model_dir, 
                                                        hparams, summary_writer)
    
    dev_scores, test_scores, global_step = run_external_eval(infer_model, infer_sess, model_dir,
                                                             hparams, summary_writer)
    
    metrics = {
        "dev_perplexity": dev_perplexity,
        "test_perplexity": test_perplexity,
        "dev_scores": dev_scores,
        "test_scores": test_scores
    }

    avg_dev_scores, avg_test_scores = None, None
    if avg_ckpts:
        avg_dev_scores, avg_test_scores = run_avg_external_eval(infer_model, infer_sess, model_dir,
                                                                hparams, summary_writer, global_step)
        metrics["avg_dev_scores"] = avg_dev_scores
        metrics["avg_test_scores"] = avg_test_scores
    
    result_summary = _format_results("dev", dev_perplexity, dev_scores, hparams.metrics)
    if avg_dev_scores:
        result_summary += ", " + _format_results("avg_dev", None, avg_dev_scores, hparams.metrics)
    
    if hparams.test_prefix:
        result_summary += ", " + _format_results("test", test_perplexity, test_scores, hparams.metrics)

        if avg_test_scores:
            result_summary += ", " + _format_results("avg_test", None, avg_test_scores, hparams.metrics)
    
    return result_summary, global_step, metrics


