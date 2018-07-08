from __future__ import absolute_import, division, print_function

import abc
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

import model_util
from utils import iterator_utils
from utils import misc_utils as utils

utils.check_tensorflow_version()

__all__ = ["BaseModel", "Model"]

# Sequence-to-sequence base class
class BaseModel(object):

    def __init__(self, hparams, iterator, mode, source_vocab_table, target_vocab_table,
                 reverse_target_vocab_table=None, scope=None, extra_args=None):
        assert isinstance(iterator, iterator_utils.BatchedInput)
        self.iterator = iterator
        self.mode = mode
        self.src_vocab_table = source_vocab_table
        self.tgt_vocab_table = target_vocab_table
        self.src_vocab_size = hparams.src_vocab_size
        self.tgt_vocab_size = hparams.tgt_vocab_size
        self.num_gpus = hparams.num_gpus
        self.time_major = hparams.time_major

        self.single_cell_fn = None
        if extra_args:
            self.single_cell_fn = extra_args.single_cell_fn
        
        self.num_encoder_layers = hparams.num_encoder_layers
        self.num_decoder_layers = hparams.num_decoder_layers
        assert self.num_encoder_layers
        assert self.num_decoder_layers

        if hasattr(hparams, "num_residual_layers"):
            self.num_encoder_residual_layers = hparams.num_residual_layers
            self.num_decoder_residual_layers = hparams.num_residual_layers
        else:
            self.num_encoder_residual_layers = hparams.num_encoder_residual_layers
            self.num_decoder_residual_layers = hparams.num_decoder_residual_layers
        
        initializer = model_util.get_initializer(hparams.init_op, hparams.random_seed, hparams.init_weight)
        tf.get_variable_scope().set_initializer(initializer)

        self.init_embeddings(hparams, scope)
        self.batch_size = tf.size(self.iterator.source_sequence_length)

        with tf.variable_scope(scope or "build_network"):
            with tf.variable_scope("decoder/output_projection"):
                self.output_layer = layers_core.Dense(hparams.tgt_vocab_size, 
                                                      use_bias=False, 
                                                      name="output_projection")
        
        res = self.build_graph(hparams, scope=scope)

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.train_loss = res[1]
            self.word_count = tf.reduce_sum(self.iterator.source_sequence_length) + tf.reduce_sum(self.iterator.target_sequence_length)
        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            self.eval_loss = res[1]
        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            self.infer_logits, _, self.final_context_state, self.sample_id = res
            self.sample_words = reverse_target_vocab_table.lookup(tf.to_int64(self.sample_id))

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            self.predict_count = tf.reduce_sum(self.iterator.target_sequence_length)
        
        self.global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()

        if self.mode ==  tf.contrib.learn.ModeKeys.TRAIN:
            self.learning_rate = tf.constant(hparams.learning_rate)
            self.learning_rate = self._get_learning_rate_warmup(hparams)
            self.learning_rate = self._get_learning_rate_decay(hparams)

            if hparams.optimizer == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                tf.summary.scalar("lr", self.learning_rate)
            elif hparams.optimizer == "adam":
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            
            gradients = tf.gradients(self.train_loss, params, colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)

            clipped_gradients, gradient_norm_summary, gradient_norm = model_util.gradient_clip(
                gradients, max_gradient_norm=hparams.max_gradient_norm)
            
            self.gradient_norm = gradient_norm
            self.update = optimizer.apply_gradients(zip(clipped_gradients, params),
                                                    global_step=self.global_step)
            
            self.train_summary = tf.summary.merge([
                tf.summary.scalar("lr", self.learning_rate),
                tf.summary.scalar("train_loss", self.train_loss)] + gradient_norm_summary)

        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            self.infer_summary = self._get_infer_summary(hparams)
        
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)

        utils.print_out("Trainable variables")
        for param in params:
            utils.print_out(" %s, %s, %s" % (param.name, str(param.get_shape()), param.op.device))


    def _get_learning_rate_warmup(self, hparams):
        warmup_steps = hparams.warmup_steps
        warmup_scheme = hparams.warmup_scheme
        utils.print_out("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" % (hparams.learning_rate, warmup_steps, warmup_scheme))

        # Inverse decay if global steps less than warmup steps
        # When step < warmup_steps,
        #   learning_rate *= warmup_factor ** (warmup_steps - global_step)
        if warmup_scheme == "t2t":
            # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
            warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
            inv_decay = warmup_factor ** (tf.to_float(warmup_steps - self.global_step))
        else:
            raise ValueError("Unknown warmup scheme %s" % warmup_scheme)
        
        return tf.cond(self.global_step < hparams.warmup_steps,
                       lambda: inv_decay * self.learning_rate,
                       lambda: self.learning_rate,
                       name="learning_rate_warmup_cond")
    

    def _get_learning_rate_decay(self, hparams):
        if hparams.decay_scheme in ["luong5", "luong10", "luong234"]:
            decay_factor = 0.5
            if hparams.decay_scheme == "luong5":
                start_decay_step = int(hparams.num_train_steps / 2)
                decay_times = 5
            elif hparams.decay_scheme == "luong10":
                start_decay_step = int(hparams.num_train_steps / 2)
                decay_times = 10
            elif hparams.decay_scheme == "luong234":
                start_decay_step = int(hparams.num_train_steps * 2 / 3)
                decay_times = 4
            remain_steps = hparams.num_train_steps - start_decay_step
            decay_steps = int(remain_steps / decay_times)
        elif not hparams.decay_scheme:
            start_decay_step = hparams.num_train_steps
            decay_steps = 0
            decay_factor = 1.0
        elif hparams.decay_scheme:
            raise ValueError("Unknown decay scheme %s" % hparams.decay_scheme)
        
        utils.print_out("decay_scheme=%s, start_decay_step=%d, decay_steps %d, decay_factor %g" % (
            hparams.decay_scheme, start_decay_step, decay_steps, decay_factor))
        
        return tf.cond(self.global_step < start_decay_step,
                       lambda: self.learning_rate,
                       lambda: tf.train.exponential_decay(self.learning_rate, 
                                                          (self.global_step - start_decay_step), 
                                                          decay_steps, decay_factor, staircase=True),
                       name="learning_rate_decay_cond")
    

    def init_embeddings(self, hparams, scope):
        self.embedding_encoder, self.embedding_decoder = model_util.create_emb_for_encoder_and_decoder(
            share_vocab=hparams.share_vocab, src_vocab_size=self.src_vocab_size, tgt_vocab_size=self.tgt_vocab_size,
            src_embed_size=hparams.num_units, tgt_embed_size=hparams.num_units, 
            num_partitions=hparams.num_embeddings_partitions, src_vocab_file=hparams.src_vocab_file,
            tgt_vocab_file=hparams.tgt_vocab_file, src_embed_file=hparams.src_embed_file,
            tgt_embed_file=hparams.tgt_embed_file, scope=scope)
    

    def train(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return sess.run([self.update, self.train_loss, self.predict_count, self.train_summary,
                         self.global_step, self.word_count, self.batch_size,
                         self.gradient_norm, self.learning_rate])
    

    def eval(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return sess.run([self.eval_loss, self.predict_count, self.batch_size])
    

    def build_graph(self, hparams, scope=None):
        utils.print_out(" creating %s graph ..." % self.mode)
        dtype = tf.float32

        with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):
            encoder_outputs, encoder_state = self._build_encoder(hparams)
            logits, sample_id, final_context_state = self._build_decoder(encoder_outputs,
                                                                         encoder_state,
                                                                         hparams)
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                with tf.device(model_util.get_device_str(self.num_encoder_layers - 1, self.num_gpus)):
                    loss = self._compute_loss(logits)
            else:
                loss = None
        return logits, loss, final_context_state, sample_id
    

    # Implemented in Model class
    @abc.abstractmethod
    def _build_encoder(self, hparams):
        pass
    

    def _build_encoder_cell(self, hparams, num_layers, num_residual_layers, base_gpu=0):
        return model_util.create_rnn_cell(unit_type=hparams.unit_type, num_units=hparams.num_units,
                                          num_layers=num_layers, num_residual_layers=num_residual_layers,
                                          forget_bias=hparams.forget_bias, dropout=hparams.dropout,
                                          mode=self.mode, num_gpus=hparams.num_gpus, base_gpu=base_gpu,
                                          single_cell_fn=self.single_cell_fn)
    

    def _get_infer_maximum_iterations(self, hparams, source_sequence_length):
        if hparams.tgt_max_len_infer:
            maximum_iterations = hparams.tgt_max_len_infer
        else:
            decoding_length_factor = 2.0
            max_encoder_length = tf.reduce_max(source_sequence_length)
            maximum_iterations = tf.to_int32(tf.round(tf.to_float(max_encoder_length) * decoding_length_factor))
        return maximum_iterations
    

    def _build_decoder(self, encoder_outputs, encoder_state, hparams):
        tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.sos)), tf.int32)
        tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.eos)), tf.int32)
        iterator = self.iterator

        maximum_iterations = self._get_infer_maximum_iterations(hparams, iterator.source_sequence_length)

        with tf.variable_scope("decoder") as decoder_scope:
            cell, decoder_initial_state = self._build_decoder_cell(hparams, encoder_outputs, 
                                                                   encoder_state, 
                                                                   iterator.source_sequence_length)
            
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                target_input = iterator.target_input
                if self.time_major:
                    target_input = tf.transpose(target_input)
                decoder_emb_input = tf.nn.embedding_lookup(self.embedding_decoder, target_input)

                helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_input, 
                                                           iterator.target_sequence_length,
                                                           time_major=self.time_major)
                
                decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state)

                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                    output_time_major=self.time_major,
                                                                                    swap_memory=True,
                                                                                    scope=decoder_scope)
                sample_id = outputs.sample_id
                # Output layer applied to all timesteps
                logits = self.output_layer(outputs.rnn_output)
            else:
                beam_width = hparams.beam_width
                length_penalty_weight = hparams.length_penalty_weight
                start_tokens = tf.fill([self.batch_size], tgt_sos_id)
                end_token = tgt_eos_id

                # Beam Search Decoding
                if beam_width > 0:
                    decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=cell, embedding=self.embedding_decoder,
                                                                   start_tokens=start_tokens, end_token=end_token,
                                                                   initial_state=decoder_initial_state, beam_width=beam_width,
                                                                   output_layer=self.output_layer,
                                                                   length_penalty_weight=length_penalty_weight)
                else:
                    sampling_temperature = hparams.sampling_temperature
                    if sampling_temperature > 0.0:
                        helper = tf.contrib.seq2seq.SampleEmbeddingHelper(self.embedding_decoder, start_tokens, end_token,
                                                                          softmax_temperature=sampling_temperature,
                                                                          seed=hparams.random_seed)
                    else:
                        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_decoder, start_tokens, end_token)
                    # Output layer applied per timestep
                    decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state, output_layer=self.output_layer)
                
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations,
                                                                                    output_time_major=self.time_major,
                                                                                    swap_memory=True,
                                                                                    scope=decoder_scope)
                if beam_width > 0:
                    logits = tf.no_op()
                    sample_id = outputs.predicted_ids
                else:
                    logits = outputs.rnn_output
                    sample_id = outputs.sample_id
        return logits, sample_id, final_context_state


    def get_max_time(self, tensor):
        time_axis = 0 if self.time_major else 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]
    

    # Implemented in Model class
    @abc.abstractmethod
    def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state, source_sequence_length):
        pass
    

    def _compute_loss(self, logits):
        target_output = self.iterator.target_output
        if self.time_major:
            target_output = tf.transpose(target_output)
        max_time = self.get_max_time(target_output)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(self.iterator.target_sequence_length, max_time, dtype=logits.dtype)
        if self.time_major:
            target_weights = tf.transpose(target_weights)
        
        loss = tf.reduce_sum(cross_entropy * target_weights) / tf.to_float(self.batch_size)
        return loss
    

    def _get_infer_summary(self, hparams):
        return tf.no_op()
    

    def infer(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        return sess.run([self.infer_logits, self.infer_summary, self.sample_id, self.sample_words])


    def decode(self, sess):
        _, infer_summary, _, sample_words = self.infer(sess)
        if self.time_major:
            sample_words = sample_words.transpose()
        elif sample_words.ndim == 3: # beam search output in [batch_size, time, beam_width] shape.
            sample_words = sample_words.transpose([2, 0, 1])
        return sample_words, infer_summary


# Sequence-to-sequence dynamic model
class Model(BaseModel):

    def _build_encoder(self, hparams):
        num_layers = self.num_encoder_layers
        num_residual_layers = self.num_encoder_residual_layers
        iterator = self.iterator

        source = iterator.source
        if self.time_major:
            source = tf.transpose(source)
        
        with tf.variable_scope("encoder") as encoder_scope:
            dtype = encoder_scope.dtype
            encoder_emb_input = tf.nn.embedding_lookup(self.embedding_encoder, source)

            if hparams.encoder_type == "uni":
                utils.print_out("num_layers = %d, num_residual_layers=%d" % (num_layers, num_residual_layers))

                cell = self._build_encoder_cell(hparams, num_layers, num_residual_layers)
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell, encoder_emb_input, dtype=dtype,
                                                                   sequence_length=iterator.source_sequence_length,
                                                                   time_major=self.time_major,
                                                                   swap_memory=True)
            elif hparams.encoder_type == "bi":
                num_bi_layers = int(num_layers / 2)
                num_bi_residual_layers = int(num_residual_layers / 2)
                utils.print_out("num_bi_layers = %d, num_bi_residual_layers=%d" % (num_bi_layers, num_bi_residual_layers))

                encoder_outputs, bi_encoder_state = self._build_bidirectional_rnn(inputs=encoder_emb_input,
                                                                               sequence_length=iterator.source_sequence_length,
                                                                               dtype=dtype,
                                                                               hparams=hparams,
                                                                               num_bi_layers=num_bi_layers,
                                                                               num_bi_residual_layers=num_bi_residual_layers)
                if num_bi_layers == 1:
                    encoder_state = bi_encoder_state
                else:
                     # alternatively concat forward and backward states
                    encoder_state = []
                    for layer in range(num_bi_layers):
                         encoder_state.append(bi_encoder_state[0][layer]) # forward
                         encoder_state.append(bi_encoder_state[1][layer]) # backward
                    encoder_state = tuple(encoder_state)
            else:
                raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)
        return encoder_outputs, encoder_state
    

    def _build_bidirectional_rnn(self, inputs, sequence_length, dtype, hparams, num_bi_layers,
                                 num_bi_residual_layers, base_gpu=0):
        forward_cell = self._build_encoder_cell(hparams, num_bi_layers, num_bi_residual_layers, base_gpu=base_gpu)
        backward_cell = self._build_encoder_cell(hparams, num_bi_layers, num_bi_residual_layers, base_gpu=(base_gpu + num_bi_layers))

        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, inputs, dtype=dtype,
                                                               sequence_length=sequence_length, 
                                                               time_major=self.time_major, swap_memory=True)
        
        return tf.concat(bi_outputs, -1), bi_state
    

    def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state, source_sequence_length):
        if hparams.attention:
            raise ValueError("BasicModel doesn't support attention")
        
        cell = model_util.create_rnn_cell(unit_type=hparams.unit_type, num_units=hparams.num_units, num_layers=self.num_decoder_layers,
                                          num_residual_layers=self.num_decoder_residual_layers, forget_bias=hparams.forget_bias,
                                          dropout=hparams.dropout, num_gpus=self.num_gpus, mode=self.mode,
                                          single_cell_fn=self.single_cell_fn)
        
        # For beam search, we need to replicate encoder infos beam_width times
        if self.mode == tf.contrib.learn.ModeKeys.INFER and hparams.beam_width > 0:
            decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=hparams.beam_width)
        else:
            decoder_initial_state = encoder_state
        
        return cell, decoder_initial_state
