from __future__ import print_function

import time
import codecs
import tensorflow as tf

from . import attention_model, model_util, model as nmt_model
from .utils import nmt_utils, misc_utils as utils

__all__ = ["load_data", "inference", "single_worker_inference", "multi_worker_instance"]


def _decode_infernce_indices(model, sess, output_infer, output_infer_summary_prefix, 
                             inference_indices, tgt_eos, subword_option):
    utils.print_out("  decoding to output %s , num sents %d." % (output_infer, len(inference_indices)))
    start_time = time.time()
    with codecs.getwriter("utf-8")(tf.gfile.GFile(output_infer, mode="wb")) as f:
        f.write("")
        for decode_id in inference_indices:
            nmt_outputs, infer_summary = model.decode(sess)

            assert nmt_outputs.shape[0] == 1
            translation = nmt_utils.get_translation(nmt_outputs, sent_id=0, tgt_eos=tgt_eos, subword_option=subword_option)

            if infer_summary is not None: # Attention model
                image_file = output_infer_summary_prefix + str(decode_id) + ".png"
                utils.print_out("attention image saved to %s*" % image_file)
                image_summary = tf.Summary()
                image_summary.ParseFromString(infer_summary)
                with tf.gfile.GFile(image_file, mode="w") as img_f:
                    img_f.write(image_summary.value[0].image_encoded_image_string)
            
            f.write("%s\n" % translation)
            utils.print_out(translation + b"\n")
    utils.print_time("done", start_time)


def load_data(inference_input_file, hparams=None):
    with codecs.getreader("utf-8")(tf.gfile.GFile(inference_input_file, mode="rb")) as f:
        inference_data = f.read().splitlines()
    
    if hparams and hparams.inference_indices:
        inference_data = [inference_data[i] for i in hparams.inference_indices]
    
    return inference_data


def inference(ckpt, inference_input_file, inference_output_file, hparams, num_workers=1, job_id=0, scope=None):
    if hparams.inference_indices:
        assert num_workers == 1
    
    if not hparams.attention:
        model_creater = nmt_model.Model
    elif hparams.attention_model == "standard":
        model_creater = attention_model.AttentionModel
    else:
        raise ValueError("Unknown model architecture")
    
    infer_model = model_util.create_infer_model(model_creater, hparams, scope)

    if num_workers == 1:
        single_worker_inference(infer_model, ckpt, inference_input_file, inference_output_file, hparams)
    else:
        multi_worker_inference(infer_model, ckpt, inference_input_file, inference_output_file, 
                               hparams, num_workers=num_workers, job_id=job_id)


def single_worker_inference(infer_model, ckpt, inference_input_file, inference_output_file, hparams):
    output_infer = inference_output_file

    infer_data = load_data(inference_input_file, hparams=hparams)

    with tf.Session(graph=infer_model.graph, config=utils.get_config_proto()) as sess:
        loaded_infer_model = model_util.load_model(infer_model.model, ckpt, sess, "infer")

        sess.run(infer_model.iterator.initializer, 
                 feed_dict={
                     infer_model.src_placeholder: infer_data,
                     infer_model.batch_size_placeholder: hparams.infer_batch_size
                 })
        utils.print_out("Start decoding")
        if hparams.inference_indices:
            _decode_infernce_indices(loaded_infer_model, sess, output_infer=output_infer,
                                     output_infer_summary_prefix=output_infer,
                                     inference_indices=hparams.inference_indices,
                                     tgt_eos=hparams.eos, subword_option=hparams.subword_option)
        else:
            nmt_utils.decode_and_evaluate("infer", loaded_infer_model, sess, output_infer, ref_file=None,
                                          metrics=hparams.metrics, subword_option=hparams.subword_option,
                                          beam_width=hparams.beam_width, tgt_eos=hparams.eos,
                                          num_translations_per_input=hparams.num_translations_per_input)


def multi_worker_inference(infer_model, ckpt, inference_input_file, inference_output_file, 
                            hparams, num_workers, job_id):
    assert num_workers > 1

    final_output_infer = inference_output_file
    output_infer = "%s_%d" % (inference_output_file, job_id)
    output_infer_done = "%s_done_%d" % (inference_output_file, job_id)

    infer_data = load_data(inference_input_file, hparams=hparams)

    # Split data to multiple workers
    total_load = len(infer_data)
    load_per_worker = int((total_load - 1) / num_workers) + 1
    start_position = job_id * load_per_worker
    end_position = min(start_position + load_per_worker, total_load)
    infer_data = infer_data[start_position: end_position]

    with tf.Session(graph=infer_model.graph, config=utils.get_config_proto()) as sess:
        loaded_infer_model = model_util.load_model(infer_model.model, ckpt, sess, "infer")

        sess.run(infer_model.iterator.initializer, 
                feed_dict={
                    infer_model.src_placeholder: infer_data,
                    infer_model.batch_size_placeholder: hparams.infer_batch_size
                })
        utils.print_out("Start decoding")
        nmt_utils.decode_and_evaluate("infer", loaded_infer_model, sess, output_infer, ref_file=None,
                                      metrics=hparams.metrics, subword_option=hparams.subword_option,
                                      beam_width=hparams.beam_width, tgt_eos=hparams.eos,
                                      num_translations_per_input=hparams.num_translations_per_input)
        
        tf.gfile.Rename(output_infer, output_infer_done, overwrite=True)
        if job_id != 0: return
        
        with codecs.getwriter("utf-8")(tf.gfile.GFile(final_output_infer, mode="wb")) as final_f:
            for worker_id in range(num_workers):
                worker_infer_done = "%s_done_%d" % (inference_output_file, worker_id)
                while not tf.gfile.Exists(worker_infer_done):
                    utils.print_out("waiting job %d to complete." % worker_id)
                    time.sleep(10)
                
                with codecs.getwriter("utf-8")(tf.gfile.GFile(worker_infer_done, mode="wb")) as f:
                    for translation in f:
                        final_f.write("%s\n" % translation)
            
            for worker_id in range(num_workers):
                worker_infer_done = "%s_done_%d" % (inference_output_file, worker_id)
                tf.gfile.Remove(worker_infer_done)
