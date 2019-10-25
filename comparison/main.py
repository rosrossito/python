from __future__ import absolute_import
from __future__ import division

import logging
import os

import tensorflow as tf

from comparison.compare import loadGloveModel, comparison_sentences

logging.basicConfig(level=logging.INFO)

MAIN_DIR = os.path.relpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # relative path of the main directory
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "comparison/data")  # relative path of data dir

# High-level options
tf.app.flags.DEFINE_integer("gpu", 0, "Which GPU to use, if you have multiple.")

# Hyperparameters
tf.app.flags.DEFINE_integer("embedding_size", 100,
                            "Size of the pretrained word vectors. This needs to be one of the available GloVe dimensions: 50/100/200/300")

# Reading and saving data
tf.app.flags.DEFINE_string("glove_path", "", "Path to glove .txt file. Defaults to data/glove.6B.{embedding_size}d.txt")
tf.app.flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR,
                           "Where to find preprocessed SQuAD data for training. Defaults to data/")

FLAGS = tf.app.flags.FLAGS

def main(unused_argv):
    # Print an error message if you've entered flags incorrectly
    if len(unused_argv) != 1:
        raise Exception("There is a problem with how you entered flags: %s" % unused_argv)

    FLAGS.glove_path = FLAGS.glove_path or os.path.join(DEFAULT_DATA_DIR,
                                                        "glove.6B.{}d.txt".format(FLAGS.embedding_size))
    loadGloveModel(FLAGS.glove_path, FLAGS.embedding_size)
    FLAGS.sentence_path = os.path.join(DEFAULT_DATA_DIR, "sentence.txt")
    FLAGS.gold_sentence_path = os.path.join(DEFAULT_DATA_DIR, "gold_sentence.txt")
    comparison_sentences(FLAGS.sentence_path, FLAGS.gold_sentence_path)

if __name__ == "__main__":
    tf.app.run()
