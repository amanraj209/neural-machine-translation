# Neural Machine Translation
Machine translation is a sub-field of computational linguistics that investigates the use of software to translate text or speech from one language to another.

Neural machine translation (NMT) is an approach to machine translation that uses a large artificial neural network to predict the likelihood of a sequence of words, typically modeling entire sentences in a single integrated model.

This repository implements a deep RNN model(Encoder - Decoder) with Attention mechanism and Beam Search decoding for langauge translation.

Model's performance is tested by BLEU scoring method.

Model is trained on small scale English-Vietnamese parallel corpus of TED talks (133K sentence pairs) provided by the IWSLT Evaluation Campaign.

## Framework Used
- [Tensorflow](https://www.tensorflow.org)

## Dataset
- Run `sh ./download_dataset.sh` to download dataset in `data/` folder.

