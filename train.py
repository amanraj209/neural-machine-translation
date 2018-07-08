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


def init_stats():
    return {
        "step_time": 0.0,
        "loss": 0.0,
        "predict_count": 0.0,
        "total_count": 0.0,
        "grad_norm": 0.0
    }


def update_stats(stats, start_time, step_result):
    (_, step_loss, step_predict_count, step_summary, global_step, step_word_count,
     batch_size, grad_norm, learning_rate) = step_result
    
    stats["step_time"] += (time.time() - start_time)
    stats["loss"] += (step_loss * batch_size)
    stats["predict_count"] += step_predict_count
    stats["total_count"] += float(step_word_count)
    stats["grad_norm"] += grad_norm

    return global_step, learning_rate, step_summary


def print_step_info(prefix, global_step, info, result_summary, log_f):
    utils.print_out("%sstep=%d learning_rate=%g step-time=%.2fs wps=%.2fK perplexity=%.2f grad_norm %.2f %s, %s" %
      (prefix, global_step, info["learning_rate"], info["avg_step_time"],
       info["speed"], info["train_perplexity"], info["avg_grad_norm"], result_summary,
       time.ctime()), log_f)


def process_stats(stats, info, global_step, steps_per_stats, log_f):
    info["avg_step_time"] = stats["step_time"] / steps_per_stats
    info["avg_grad_norm"] = stats["grad_norm"] / steps_per_stats
    info["train_perplexity"] = utils.safe_exp(stats["loss"] / stats["predict_count"])
    info["speed"] = stats["total_count"] / (stats["step_time"] * 1000)

    is_overflow = False
    train_perplexity = info["train_perplexity"]
    if math.isnan(train_perplexity) or math.isinf(train_perplexity) or train_perplexity > 1e20:
        utils.print_out("step %d overflow, stop early" % global_step, log_f)
        is_overflow = True
    return is_overflow


# Tasks to do before training
def before_train(loaded_train_model, train_model, train_sess, global_step, hparams, log_f):
    stats = init_stats()
    info = {
        "train_perplexity": 0.0,
        "speed": 0.0,
        "avg_step_time": 0.0,
        "avg_grad_norm": 0.0,
        "learning_rate": loaded_train_model.learning_rate.eval(session=train_sess)
    }
    start_train_time = time.time()
    utils.print_out("start_step=%d, learning_rate=%g, %s" % (global_step, info["learning_rate"], time.ctime()), log_f)
    
    skip_count = hparams.batch_size * hparams.epoch_step
    utils.print_out("Init train iterator, skipping %d elements" % skip_count)
    train_sess.run(train_model.iterator.initializer, feed_dict={train_model.skip_count_placeholder: skip_count})
    return stats, inference, start_train_time


# Summary of the current best results
def _get_best_results(hparams):
    tokens = []
    for metric in hparams.metrics:
        tokens.append("%s %.2f" % (metric, getattr(hparams, "best_" + metric)))
    return ", ".join(tokens)


# Train a translation model
def train(hparams, scope=None, target_session=""):
    log_device_placement = hparams.log_device_placement
    out_dir = hparams.out_dir
    num_train_steps = hparams.num_train_steps
    steps_per_stats = hparams.steps_per_stats
    steps_per_external_eval = hparams.steps_per_external_eval
    steps_per_eval = steps_per_stats * 10
    avg_ckpts = hparams.avg_ckpts

    if not steps_per_external_eval:
        steps_per_external_eval = steps_per_eval * 5
    
    if not hparams.attention:
        model_creator = nmt_model.Model
    else:
        if hparams.attention_architecture == "standard":
            model_creator = attention_model.AttentionModel
        else:
            raise ValueError("Unknown attention architecture %s" % hparams.attention_architecture)
    
    train_model = model_util.create_train_model(model_creator, hparams, scope)
    eval_model = model_util.create_eval_model(model_creator, hparams, scope)
    infer_model = model_util.create_infer_model(model_creator, hparams, scope)

    dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
    dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
    sample_src_data = inference.load_data(dev_src_file)
    sample_tgt_data = inference.load_data(dev_tgt_file)

    summary_name = "train_log"
    model_dir = hparams.out_dir

    log_file = os.path.join(out_dir, "log_%d" % time.time())
    log_f = tf.gfile.GFile(log_file, mode="a")
    utils.print_out("log_file=%s" % log_file, log_f)

    config_proto = utils.get_config_proto(
        log_device_placement=log_device_placement,
        num_intra_threads=hparams.num_intra_threads,
        num_inter_threads=hparams.num_inter_threads
    )

    train_sess = tf.Session(target=target_session, config=config_proto, graph=train_model.graph)
    eval_sess = tf.Session(target=target_session, config=config_proto, graph=eval_model.graph)
    infer_sess = tf.Session(target=target_session, config=config_proto, graph=infer_model.graph)

    with train_model.graph.as_default():
        loaded_train_model, global_step = model_util.create_or_load_model(train_model.model, model_dir, train_sess, "train")

    summary_writer = tf.summary.FileWriter(os.path.join(out_dir, summary_name), train_model.graph)

    run_full_eval(model_dir, infer_model, infer_sess, eval_model, eval_sess, hparams, summary_writer, sample_src_data, sample_tgt_data, avg_ckpts)
    last_stats_step = global_step
    last_eval_step = global_step
    last_external_eval_step = global_step

    stats, info, start_train_time = before_train(loaded_train_model, train_model, train_sess, global_step, hparams, log_f)
    while global_step < num_train_steps:
        start_time = time.time()
        try:
            step_result = loaded_train_model.train(train_sess)
            hparams.epoch_step += 1
        except tf.errors.OutOfRangeError:
            hparams.epoch_step = 0
            utils.print_out("finished an epoch, step %d. Perform external evaluation" % global_step)
            run_sample_decode(infer_model, infer_sess, model_dir, hparams, summary_writer, sample_src_data, sample_tgt_data)
            run_external_eval(infer_model, infer_sess, model_dir, hparams, summary_writer)

            if avg_ckpts:
                run_avg_external_eval(infer_model, infer_sess, model_dir, hparams, summary_writer, global_step)

            train_sess.run(train_model.iterator.initializer, feed_dict={train_model.skip_count_placeholder: 0})
            continue
        
        global_step, info["learning_rate"], step_summary = update_stats(stats, start_time, step_result)
        summary_writer.add_summary(step_summary, global_step)

        if global_step - last_stats_step >= steps_per_stats:
            last_stats_step = global_step
            is_overflow = process_stats(stats, info, global_step, steps_per_stats, log_f)
            print_step_info("  ", global_step, info, _get_best_results(hparams), log_f)
            if is_overflow:
                break
            stats = init_stats()
        
        if global_step - last_eval_step >= steps_per_eval:
            last_eval_step = global_step
            utils.print_out("save eval, global step %d" % global_step)
            utils.add_summary(summary_writer, global_step, "train_perplexity", info["train_perplexity"])

            loaded_train_model.saver.save(train_sess, os.path.join(out_dir, "translate.ckpt"), global_step=global_step)

            run_sample_decode(infer_model, infer_sess, model_dir, hparams, summary_writer, sample_src_data, sample_tgt_data)
            run_internal_eval(eval_model, eval_sess, model_dir, hparams, summary_writer)

        if global_step - last_external_eval_step >= steps_per_external_eval:
            last_external_eval_step = global_step
            loaded_train_model.saver.save(train_sess, os.path.join(out_dir, "translate.ckpt"), global_step=global_step)

            run_sample_decode(infer_model, infer_sess, model_dir, hparams, summary_writer, sample_src_data, sample_tgt_data)
            run_external_eval(infer_model, infer_sess, model_dir, hparams, summary_writer)

            if avg_ckpts:
                run_avg_external_eval(infer_model, infer_sess, model_dir, hparams, summary_writer, global_step)

    loaded_train_model.saver.save(train_sess, os.path.join(out_dir, "translate.ckpt"), global_step=global_step)

    (result_summary, _, final_eval_metrics) = run_full_eval(model_dir, infer_model, infer_sess, eval_model,
                                                            eval_sess, hparams, summary_writer, sample_src_data,
                                                            sample_tgt_data, avg_ckpts)    
    print_step_info("final, ", global_step, info, result_summary, log_f)
    utils.print_time("done training!", start_train_time)    

    summary_writer.close()

    utils.print_out("# Start evaluating saved best models")
    for metric in hparams.metrics:
        best_model_dir = getattr(hparams, "best_" + metric + "_dir")
        summary_writer = tf.summary.FileWriter(os.path.join(best_model_dir, summary_name), infer_model.graph)
        result_summary, best_global_step, _ = run_full_eval(best_model_dir, infer_model, infer_sess,
                                                         eval_model, eval_sess, hparams, summary_writer,
                                                         sample_src_data, sample_tgt_data)
        print_step_info("best %s, " % metric, best_global_step, info, result_summary, log_f)
        summary_writer.close()

        if avg_ckpts:
            best_model_dir = getattr(hparams, "avg_best_" + metric + "_dir")
            summary_writer = tf.summary.FileWriter(os.path.join(best_model_dir, summary_name), infer_model.graph)
            result_summary, best_global_step, _ = run_full_eval(best_model_dir, infer_model, infer_sess,
                                                             eval_model, eval_sess, hparams, summary_writer,
                                                             sample_src_data, sample_tgt_data)
            print_step_info("avg_best %s, " % metric, best_global_step, info, result_summary, log_f)
            summary_writer.close()
    
    return final_eval_metrics, global_step

