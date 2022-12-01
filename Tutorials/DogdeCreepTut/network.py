import tensorflow as tf
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model, kernel_regularizer='l2')
    self.wk = tf.keras.layers.Dense(d_model, kernel_regularizer='l2')
    self.wv = tf.keras.layers.Dense(d_model, kernel_regularizer='l2')

    self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout = tf.keras.layers.Dropout(0.1)
    
    #v = tf.Variable(tf.random.truncated_normal([10, 40]))
    # W = tf.Variable(tf.random_uniform([16,4],0,0.01))
    #self.w = tf.Variable(tf.random.truncated_normal([64, 1, 100]), 0.0, 0.1)
    
    self.dense = tf.keras.layers.Dense(d_model, kernel_regularizer='l2')

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'd_model': self.d_model,
        'num_heads': self.num_heads,
    })
    return config
    
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask, training):
    batch_size = tf.shape(q)[0]
    
    v_original = v
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    output = self.dense(output) 
    
    return output, attention_weights


class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""
  def __init__(
      self, 
      num_actions: int, 
      num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.num_actions = num_actions
    
    #self.conv_1 = layers.Conv2D(16, 8, 4, padding="valid", activation="relu", kernel_regularizer='l2')
    #self.conv_2 = layers.Conv2D(32, 4, 2, padding="valid", activation="relu", kernel_regularizer='l2')
    #self.conv_3 = layers.Conv2D(64, 3, 1, padding="valid", activation="relu", kernel_regularizer='l2')
    self.dense_a = layers.Dense(1024, activation='relu')
    self.dense_b = layers.Dense(1024, activation='relu')
    self.dense_c = layers.Dense(1024, activation='relu')
    
    #self.attention = MultiHeadAttention(33, 3)
    #self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    #self.dropout = tf.keras.layers.Dropout(0.1)

    self.lstm = layers.LSTM(128, return_sequences=True, return_state=True, kernel_regularizer='l2')
    
    self.common = layers.Dense(num_hidden_units, activation="relu", kernel_regularizer='l2')
    self.actor = layers.Dense(num_actions, kernel_regularizer='l2')
    self.critic = layers.Dense(1, kernel_regularizer='l2')

    #self.conv_out_size = 7
    #self.locs = []
    #for i in range(0, self.conv_out_size * self.conv_out_size):
    #    self.locs.append(i / float(self.conv_out_size * self.conv_out_size))
        
    #self.locs = tf.expand_dims(self.locs, 0)
    #self.locs = tf.expand_dims(self.locs, 2)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'num_actions': self.num_actions,
        'num_hidden_units': self.num_hidden_units
    })

    return config
    
  def call(self, inputs: tf.Tensor, memory_state: tf.Tensor, carry_state: tf.Tensor, training) -> Tuple[tf.Tensor, tf.Tensor, 
                                                                                                        tf.Tensor, tf.Tensor]:
    batch_size = tf.shape(inputs)[0]

    inputs = layers.Flatten()(inputs)
        
    dense_a = self.dense_a(inputs)
    dense_b = self.dense_b(dense_a)
    dense_c = self.dense_c(dense_b)

    #print("inputs.shape: ", inputs.shape)

    #conv_1 = self.conv_1(inputs)
    #conv_2 = self.conv_2(conv_1)
    #conv_3 = self.conv_3(conv_2)
    #print("conv_3.shape: ", conv_3.shape)
    #conv_3_reshaped = layers.Reshape((64,16))(conv_3)
    
    #conv_3_features = tf.reshape(conv_3, [batch_size,self.conv_out_size*self.conv_out_size,32])
        
    #locs = tf.tile(self.locs, [batch_size, 1, 1])
    #conv_3_features_locs = tf.concat([conv_3_features, locs], 2)
    #print("conv_3_features_locs.shape: ", conv_3_features_locs.shape)

    #attention_output, _ = self.attention(conv_3_features_locs, conv_3_features_locs, conv_3_features_locs, None)
    #attention_output = self.dropout(attention_output, training=training)
    #attention_output = self.layernorm(conv_3_features_locs + attention_output)
    
    #print("attention_output.shape: ", attention_output.shape)

    #max_pool_1d = tf.math.reduce_max(attention_output, 1)
    #print("max_pool_1d: ", max_pool_1d)

    #X_input = layers.Flatten()(inputs)
    #dense_1 = self.dense_1(X_input)
    dense_c_reshaped = layers.Reshape((64,16))(dense_c)

    initial_state = (memory_state, carry_state)
    lstm_output, final_memory_state, final_carry_state  = self.lstm(dense_c_reshaped, initial_state=initial_state, 
                                                                    training=training)
    
    outputs = layers.Flatten()(lstm_output)
    #print("X_input.shape: ", X_input.shape)
    x = self.common(outputs)
    
    return self.actor(x), self.critic(x), final_memory_state, final_carry_state