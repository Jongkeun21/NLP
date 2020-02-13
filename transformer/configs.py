import tensorflow as tf

tf.app.flags.DEFINE_string('f', '', 'kernel')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('train_steps', 20000, 'train steps')
tf.app.flags.DEFINE_float('dropout_width', 0.5, 'dropout width')
tf.app.flags.DEFINE_integer('embedding_size', 128, 'embedding size')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
tf.app.flags.DEFINE_integer('shuffle_seek', 1000, 'shuffle random seek')
tf.app.flags.DEFINE_integer('max_sequence_length', 25, 'max sequence length')
tf.app.flags.DEFINE_integer('model_hidden_size', 128, 'model weights size')
tf.app.flags.DEFINE_integer('ffn_hidden_size', 128, 'ffn weights size')
tf.app.flags.DEFINE_integer('attention_head_size', 4, 'attn head size')
tf.app.flags.DEFINE_integer('layer_size', 2, 'layer_size')
tf.app.flags.DEFINE_string('data_path', './data_in/ChatBotData.csv', 'data path')
tf.app.flags.DEFINE_string('vocabulary_path', './data_out/vocabularyData.voc', 'vocabulary path')
tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point', 'check point path')
tf.app.flags.DEFINE_boolean('tokenize_as_morph', False, 'set morph tokenize')

DEFINES = tf.app.flags.FLAGS