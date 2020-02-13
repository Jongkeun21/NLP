import tensorflow as tf

tf.app.flags.DEFINE_string('f', '', 'kernel')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('train_steps', 20000, 'train steps')
tf.app.flags.DEFINE_float('dropout_width', 0.8, 'dropout width')
tf.app.flags.DEFINE_integer('layer_size', 1, 'layer size')
tf.app.flags.DEFINE_integer('hidden_size', 128, 'weights size')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
tf.app.flags.DEFINE_float('teacher_forcing_rate', 0.7, 'teacher forcing rate')
tf.app.flags.DEFINE_string('data_path', './data_in/ChatBotData.csv', 'data path')
tf.app.flags.DEFINE_string('vocab_path', './data_out/vocabularyData.voc', 'vocab path')
tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point', 'check point path')
tf.app.flags.DEFINE_string('save_model_path', './data_out/model', 'save model')
tf.app.flags.DEFINE_integer('shuffle_seek', 1000, 'shuffle random seek')
tf.app.flags.DEFINE_integer('max_seq_length', 25, 'max sequence length')
tf.app.flags.DEFINE_integer('embedding_size', 128, 'embedding size')
tf.app.flags.DEFINE_boolean('embedding', True, 'Use Embedding flag')
tf.app.flags.DEFINE_boolean('multilayer', True, 'Use Multi RNN cell')
tf.app.flags.DEFINE_boolean('attention', True, 'Use Attention')
tf.app.flags.DEFINE_boolean('teacher_forcing', True, 'Use Teacher Forcing')
tf.app.flags.DEFINE_boolean('tokenize_as_morph', False, 'set morph tokenize')
tf.app.flags.DEFINE_boolean('serving', False, 'Use Serving')
tf.app.flags.DEFINE_boolean('loss_mask', False, 'Use loss mask')

DEFINES = tf.app.flags.FLAGS