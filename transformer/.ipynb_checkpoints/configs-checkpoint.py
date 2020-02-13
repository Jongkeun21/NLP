{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "\n",
    "tf.app.flags.DEFINE_string('f', '', 'kernel')\n",
    "tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')\n",
    "tf.app.flags.DEFINE_integer('train_steps', 20000, 'train steps')\n",
    "tf.app.flags.DEFINE_float('dropout_width', 0.5, 'dropout width')\n",
    "tf.app.flags.DEFINE_integer('embedding_size', 128, 'embedding size')\n",
    "tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')\n",
    "tf.app.flags.DEFINE_integer('shuffle_seek', 1000, 'shuffle random seek')\n",
    "tf.app.flags.DEFINE_integer('max_sequence_length', 25, 'max sequence length')\n",
    "tf.app.flags.DEFINE_integer('model_hidden_size', 128, 'model weights size')\n",
    "tf.app.flags.DEFINE_integer('ffn_hidden_size', 128, 'ffn weights size')\n",
    "tf.app.flags.DEFINE_integer('attention_head_size', 4, 'attn head size')\n",
    "tf.app.flags.DEFINE_integer('layer_size', 2, 'layer_size')\n",
    "tf.app.flags.DEFINE_string('data_path', './data_in/ChatBotData.csv', 'data path')\n",
    "tf.app.flags.DEFINE_string('vocabulary_path', './data_out/vocabularyData.voc', 'vocabulary path')\n",
    "tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point', 'check point path')\n",
    "tf.app.flags.DEFINE_boolean('tokenize_as_morph', False, 'set morph tokenize')\n",
    "\n",
    "DEFINES = tf.app.flags.FLAGS"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
