import re
import os
import codecs
import subprocess
import tensorflow as tf

from scripts import bleu, rouge

__all__ = ["evaluate"]

def evaluate(ref_file, trans_file, metric, subword_option=None):
    # BLEU scores for translation task
    if metric.lower() == "bleu":
        evaluation_score = _bleu(ref_file, trans_file, subword_option=subword_option)
    # ROUGE scores for summarization tasks
    elif metric.lower() == "rouge":
        evaluation_score = _rouge(ref_file, trans_file, subword_option=subword_option)
    elif metric.lower() == "accuracy":
        evaluation_score = _accuracy(ref_file, trans_file)
    elif metric.lower() == "word_accuracy":
        evaluation_score = _word_accuracy(ref_file, trans_file)
    else:
        raise ValueError("Unknown metric %s" % metric)

    return evaluation_score


def _clean(sentence, subword_option):
    sentence = sentence.strip()
    # BPE (Byte Pair Encoding)
    if subword_option == "bpe":
        sentence = re.sub("@@ ", "", sentence)
    # SPM
    elif subword_option == "spm":
        sentence = u"".join(sentence.split()).replace(u"\u2581", u" ").lstrip()
    return sentence


# BLEU scores for translation task
def _bleu(ref_file, trans_file, subword_option=None):
    max_order = 4
    smooth = False

    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with codecs.getreader("utf-8")(tf.gfile.GFile(reference_filename, "rb")) as f:
            reference_text.append(f.readlines())

    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference = _clean(reference, subword_option)
            reference_list.append(reference.split(" "))
        per_segment_references.append(reference_list)

    translations = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as f:
        for line in f:
            line = _clean(line, subword_option=None)
            translations.append(line.split(" "))

    bleu_score, _, _, _, _, _ = bleu.compute_bleu(per_segment_references, translations, max_order, smooth)
    return 100 * bleu_score


# ROUGE scores for summarization tasks
def _rouge(ref_file, summarization_file, subword_option=None):
    references = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(ref_file, "rb")) as f:
        for line in f:
            references.append(_clean(line, subword_option))

    hypotheses = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(summarization_file, "rb")) as f:
        for line in f:
            hypotheses.append(_clean(line, subword_option=None))

    rouge_score_map = rouge.rouge(hypotheses, references)
    return 100 * rouge_score_map["rouge_l/f_score"]


def _accuracy(label_file, pred_file):
    with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "rb")) as label_f:
        with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "rb")) as pred_f:
            count = 0.0
            match = 0.0
            for label in label_f:
                label = label.strip()
                pred = pred_f.readline().strip()
                if label == pred:
                    match += 1
                count += 1
    return 100 * match / count


def _word_accuracy(label_file, pred_file):
    with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "r")) as label_f:
        with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "r")) as pred_f:
            total_acc, total_count = 0., 0.
            for sentence in label_f:
                labels = sentence.strip().split(" ")
                preds = pred_f.readline().strip().split(" ")
                match = 0.0
                for pos in range(min(len(labels), len(preds))):
                    label = labels[pos]
                    pred = preds[pos]
                    if label == pred:
                        match += 1
                total_acc += 100 * match / max(len(labels), len(preds))
                total_count += 1
    return total_acc / total_count
