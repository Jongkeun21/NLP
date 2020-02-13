{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import sys\n",
    "import numpy as np\n",
    "from configs import DEFINES\n",
    "\n",
    "def layer_norm(inputs, eps=1e-6) :\n",
    "    feature_shape = inputs.get_shape()[-1:]\n",
    "    mean = tf.keras.backend.mean(inputs, [-1], keepdims=True)\n",
    "    std = tf.keras.backend.std(inputs, [-1,], keepdims=True)\n",
    "    beta = tf.get_variable(\"beta\", initializer=tf.zeros(feature_shape))\n",
    "    gamma = tf.get_variable(\"gamma\", initializer=tf.ones(feature_shape))\n",
    "    \n",
    "    return gamma*(inputs-mean)/(std+eps)+beta\n",
    "\n",
    "def sublayer_connection(inputs, sublayer, dropout=0.2) :\n",
    "    outputs = layer_norm(inputs, tf.keras.layers.Dropout(dropout)(sublayer))\n",
    "    \n",
    "    return outputs\n",
    "\n",
    "def positional_encoding(dim, sentence_length) :\n",
    "    encoded_vec = np.array([pos/np.power(1000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])\n",
    "    encoded_vec[::2] = np.sin(encoded_vec[::2])\n",
    "    encoded_vec[1::2] = np.cos(encoded_vec[1::2])\n",
    "    \n",
    "    return tf.constant(encoded_vec.reshape([sentence_length, dim]), dtype=tf.float32)\n",
    "\n",
    "class MultiHeadAttention(tf.keras.Model) :\n",
    "    def __init__(self, num_units, heads, masked=False) :\n",
    "        super(MultiHeadAttention, self).__init__()\n",
    "        \n",
    "        self.heads = heads\n",
    "        self.masked = masked\n",
    "        \n",
    "        self.query_dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)\n",
    "        self.key_dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)\n",
    "        self.value_dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)\n",
    "     \n",
    "    # Scaled dot-product Attention\n",
    "    def scaled_dot_product_attention(self, query, key, value, masked=False) :\n",
    "        key_seq_length = float(key.get_shape().as_list()[-1])\n",
    "        key = tf.transpose(key, perm=[0, 2, 1])\n",
    "        outputs = tf.matmul(query, key)/tf.sqrt(key_seq_length)\n",
    "\n",
    "        # Masking 과정\n",
    "        if masked :\n",
    "            diag_vals = tf.ones_like(outputs[0, :, :])\n",
    "            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()\n",
    "            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])\n",
    "\n",
    "            paddings = tf.ones_like(masks)*(-2**32+1)\n",
    "            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)\n",
    "\n",
    "        attention_map = tf.nn.softmax(outputs)\n",
    "\n",
    "        return tf.matmul(attention_map, value)\n",
    "\n",
    "    # Multi-head Attention 연산\n",
    "    def call(self, query, key, value) :\n",
    "        query = self.query_dense(query)\n",
    "        key = self.key_dense(key)\n",
    "        value = self.value_dense(value)\n",
    "        \n",
    "        query = tf.concat(tf.split(query, self.heads, axis=-1), axis=0)\n",
    "        key = tf.concat(tf.split(key, self.heads, axis=-1), axis=0)\n",
    "        value = tf.concat(tf.split(value, self.heads, axis=-1), axis=0)\n",
    "        \n",
    "        attention_map = self.scaled_dot_product_attention(query, key, value, self.masked)\n",
    "        attn_outputs = tf.concat(tf.split(attention_map, self.heads, axis=0), axis=-1)\n",
    "        \n",
    "        return attn_outputs\n",
    "\n",
    "class PositionWiseFeedForward(tf.keras.Model) :\n",
    "    def __init__(self, num_units, feature_shape) :\n",
    "        super(PositionWiseFeedForward, self).__init__()\n",
    "        \n",
    "        self.inner_dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)\n",
    "        self.output_dense = tf.keras.layers.Dense(feature_shape)\n",
    "        \n",
    "    def call(self, inputs) :\n",
    "        inner_layer = self.inner_dense(inputs)\n",
    "        outputs = self.output_dense(inner_layer)\n",
    "        \n",
    "        return outputs\n",
    "\n",
    "class Encoder(tf.keras.Model) :\n",
    "    def __init__(self, model_dims, ffn_dims, attn_heads, num_layers=1) :\n",
    "        super(Encoder, self).__init__()\n",
    "        \n",
    "        self.self_attention = [MultiHeadAttention(model_dims, attn_heads) for _ in range(num_layers)]\n",
    "        self.position_feedforward = [PositionWiseFeedForward(ffn_dims, model_dims) for _ in range(num_layers)]\n",
    "        \n",
    "    def call(self, inputs) :\n",
    "        output_layer = None\n",
    "        \n",
    "        for i, (s_a, p_f) in enumerate(zip(self.self_attention, self.position_feedforward)) :\n",
    "            with tf.variable_scope('encoder_layer_' + str(i+1)) :\n",
    "                attention_layer = sublayer_connection(inputs, s_a(inputs, inputs, inputs))\n",
    "                output_layer = sublayer_connection(attention_layer, p_f(attention_layer))\n",
    "                \n",
    "                inputs = output_layer\n",
    "                \n",
    "        return output_layer\n",
    "\n",
    "class Decoder(tf.keras.Model) :\n",
    "    def __init__(self, model_dims, ffn_dims, attn_heads, num_layers=1) :\n",
    "        super(Decoder, self).__init__()\n",
    "        \n",
    "        self.self_attention = [MultiHeadAttention(model_dims, attn_heads, masked=True) for _ in range(num_layers)]\n",
    "        self.encoder_decoder_attention = [MultiHeadAttention(model_dims, attn_heads) for _ in range(num_layers)]\n",
    "        self.position_feedforward = [MultiHeadAttention(ffn_dims, model_dims) for _ in range(num_layers)]\n",
    "        \n",
    "    def call(self, inputs, encoder_outputs) :\n",
    "        output_layer = None\n",
    "        \n",
    "        for i, (s_a, ed_a, p_f) in enumerate(zip(self.self_attention, self.encoder_decoder_attention, self.position_feedforward)) :\n",
    "            with tf.variable_scope('decoder_layer_' + str(i+1)) :\n",
    "                masked_attetion_layer = sublayer_connection(inputs, s_a(inputs, inputs, inputs))\n",
    "                attention_layer = sublayer_connection(masked_attetion_layer, ed_a(masked_attetion_layer, encoder_outputs, encoder_outputs))\n",
    "                output_layer = sublayer_connection(attention_layer, p_f(attention_layer))\n",
    "                inputs = output_layer\n",
    "                \n",
    "        return output_layer\n",
    "\n",
    "def Model(features, labels, mode, params) :\n",
    "    TRAIN = mode = tf.estimator.ModeKeys.TRAIN\n",
    "    EVAL = mode = tf.estimator.ModeKeys.EVAL\n",
    "    PREDICT = mode = tf.estimator.ModeKeys.PREDICT\n",
    "    \n",
    "    position_encode = positional_encoding(params['embedding_size'], params['max_sequence_length'])\n",
    "    embedding = tf.keras.layers.Embedding(params['vocabulary_length'], params['embedding_size'])\n",
    "    \n",
    "    encoder_layers = Encoder(params['model_hidden_size'], params['ffn_hidden_size'], params['attention_hidden_size'], params['layer_size'])\n",
    "    decoder_layers = Decoder(params['model_hidden_size'], params['ffn_hidden_size'], params['attention_hidden_size'], params['layer_size'])\n",
    "    logit_layer = tf.keras.layers.Dense(params['vocabulary_length'])\n",
    "    \n",
    "    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE) :\n",
    "        x_embedded_matrix = embedding(features['input']) + position_encode\n",
    "        encoder_outputs = encoder_layers(x_embedded_matrix)\n",
    "        \n",
    "    loop_count = params['max_sequence_length'] if PREDICT else 1\n",
    "    \n",
    "    predict, output, logits = None, None, None\n",
    "    \n",
    "    for i in range(loop_count) :\n",
    "        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE) :\n",
    "            if i > 0 :\n",
    "                output = tf.concat([tf.ones((output.shape[0], 1), dtype=tf.int64), predict[:, :-1]], axis=-1)\n",
    "            else :\n",
    "                output = features['output']\n",
    "                \n",
    "            y_embedded_matrix = embedding(output) + position_encode\n",
    "            decoder_outputs = decoder_layers(y_embedded_matrix, encoder_outputs)\n",
    "            \n",
    "            logits = logit_layer(decoder_outputs)\n",
    "            predict = tf.argmax(logits, 2)\n",
    "            \n",
    "    if PREDICT :\n",
    "        predictions = {\n",
    "            'indexs': predict,\n",
    "            'logits': logits\n",
    "        }\n",
    "        \n",
    "        return tf.estimator.EstimatorSpec(mode, predictions=predictions)\n",
    "    \n",
    "    loss = tf.reduce_mean(tf.nn.sparse_softmax_corss_entropy_with_logits(logits=logits, labels=labels))\n",
    "    accuracy = tf.metrics.accuracy(labels=labels, predictions=predict, name='accOp')\n",
    "    \n",
    "    metrics = {'accuracy': accuracy}\n",
    "    tf.summary.scalar('accuracy', accuracy[1])\n",
    "    \n",
    "    if EVAL :\n",
    "        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)\n",
    "    \n",
    "    assert TRAIN\n",
    "    \n",
    "    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])\n",
    "    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())\n",
    "    \n",
    "    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=traion_op)"
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