from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import tensorflow as tf

FLAGS = None
os.environ['KMP_DUPLICATE_LIB_OK']='True' #ONLY A MAC PROBLEM??????

def train():

  np.array  
    
  #------Data import and general setup------
  n_steps=FLAGS.n_steps
  n_inputs=13
  batch_size=FLAGS.batch_size
  n_neurons=FLAGS.n_neurons
  n_fc_layers=FLAGS.n_layers
  nn_type=FLAGS.nn_type.strip()
  n_epochs=FLAGS.n_epochs
  n_class=61
  n_lstm_layers=FLAGS.n_layers
  
  ##FIX THIS
  def normalize_data(mat, mat_val):
    mean = np.mean(mat, axis=0)
    std = np.std(mat, axis=0)
    return ((mat-mean) / std, (mat_val-mean) / std)
    
  #Import numpy data
  #Import training data
  features = np.load("train_mfcc.npy")
  features = features.astype(np.dtype('f4'))
  
  labels = np.load("train_labels.npy")
  
  validation_features = np.load("test_mfcc.npy")
  validation_features = validation_features.astype(np.dtype('f4')) 
  
  
  features, validation_features = normalize_data(features, validation_features)
  
  validation_labels = np.load("test_labels.npy")  
  #Transforms labels into one-hot vectors
  #labels = tf.one_hot(indices=labels, depth=n_class, on_value=1, off_value=0)

  #Check if length of data and lables are the same 
  assert features.shape[0] == labels.shape[0]
  assert validation_features.shape[0] == validation_labels.shape[0]

  #Create the dataset class from the loaded numpy tensors and generates batches
  
  features_ph = tf.placeholder(features.dtype, features.shape)
  labels_ph = tf.placeholder(labels.dtype, labels.shape)
  
  validation_features_ph = tf.placeholder(validation_features.dtype, validation_features.shape)
  validation_labels_ph = tf.placeholder(validation_labels.dtype, validation_labels.shape)
  
  dataset = tf.data.Dataset.from_tensor_slices((features_ph, labels_ph))
  validation_dataset = tf.data.Dataset.from_tensor_slices((validation_features_ph, validation_labels_ph))


  if nn_type == 'dense' or nn_type == 'lstm':
    print('Your ANN is valid')
    if nn_type == 'dense':  
      print("You have sleected dense")
      dataset = dataset.batch(batch_size, drop_remainder=True)
      validation_dataset = validation_dataset.batch(500, drop_remainder=True)    
      iterator = dataset.make_initializable_iterator()
      validation_iterator = validation_dataset.make_initializable_iterator()
    if nn_type == 'lstm':
      print("You have sleected lstm")
      dataset = dataset.batch(batch_size*n_steps, drop_remainder=True)
      validation_dataset = validation_dataset.batch(500*n_steps, drop_remainder=True)  
      iterator = dataset.make_initializable_iterator()
      validation_iterator = validation_dataset.make_initializable_iterator()
  else:
    print("Your ANN is not valid")
    raise Exception('Select a valid neural network')
    return


  #This is needed for jupyter notebooks, if i'm not mistaken
  sess = tf.InteractiveSession()

  #------Definition of functions of layers which will be used in this model------
  #Simple fully connected layer
  def FC_layer(input_tensor, output_dim, layer_name, act=tf.nn.relu):
    output_tensor = tf.layers.dense(input_tensor, output_dim, name=layer_name, activation=act)
    tf.summary.histogram('fc',output_tensor)
    return output_tensor

  def MultiFC_layer(input_tensor, n_neurons, n_layers, layer_name, act=tf.nn.relu):
      fc_layers = []
      layer=input_tensor
      for i in range(n_layers):
          name = "%s_%s" % (layer_name, i)
          layer = FC_layer(layer, n_neurons, name, act=act)
          fc_layers.append(layer)
      return fc_layers[n_layers-1]


  #Simple LSTM layer which returns the outputs
  def LSTM_layer(input_tensor, n_neurons, layer_name):
    LSTM_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_neurons, name='LSTM_cell')
    outputs, states = tf.nn.dynamic_rnn(LSTM_cell, input_tensor, dtype=tf.float32)
    tf.summary.histogram('states', states)
      #returns last output state of the sequence
    return outputs

  #Dropout layer
  def drp_layer(input_tensor, layer_name):
    dropped = tf.nn.dropout(input_tensor, keep_prob, name=layer_name)
    return dropped

  def MultiLSTM_layer(input_tensor, subsequence_length, n_neurons, n_layers, initial_state, keep_prob):
    dropout_cell = []
    for i in range (n_layers):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_neurons)
        dropout_cell.append( tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob))
    cell = tf.nn.rnn_cell.MultiRNNCell(dropout_cell)
    rnn_out, rnn_state = tf.nn.dynamic_rnn(cell, input_tensor, initial_state=initial_state, dtype=tf.float32)
    return rnn_out, rnn_state

  #------Neural networks description------  
  ###### FC net description ###### 
  if nn_type=='dense':  
    
      ###### Input placeholders######
      with tf.name_scope('Input'):
        X = tf.placeholder(tf.float32, shape=[None, n_inputs], name='X-input')
        y_ = tf.placeholder(tf.int64, [None], name='y-input')
        
      #Dropout probability placeholder
      keep_prob = tf.placeholder(tf.float32, name='keep_prob')
      tf.summary.scalar('keep_probability', keep_prob)
      
      #initial_state = tf.placeholder(tf.float32, [None, None], name='initial_state')
      
      ###### Graph definition

      fc_net = MultiFC_layer(X, n_neurons, n_fc_layers, 'FC')          
                               
      #FC_1=FC_layer(X, FLAGS.n_neurons, 'FC_1')
      #FC_2=FC_layer(FC_1, FLAGS.n_neurons, 'FC_2')
      #FC_3=FC_layer(FC_2, FLAGS.n_neurons, 'FC_3')
    
    
      #Output layer, activation identity!!
      y = FC_layer(fc_net, n_class, 'output_FC', act=tf.identity)
    
      #Optimizer and gradient descent definition
      #Sparse cross entropy as loss function
      with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
          cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
    
      tf.summary.scalar('cross_entropy', cross_entropy)
    
      #Optimizer definition (keep ADAM)
      with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    
      #Validation and accuracy computation
      with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
          correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        with tf.name_scope('accuracy'):
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
          tf.summary.scalar('accuracy', accuracy) 
      
    
  ###### LSTM net description ######
  if nn_type=='lstm':

      ###### Input placeholders######
      with tf.name_scope('Input'):
        X = tf.placeholder(tf.float32, shape=[None, n_inputs], name='X-input')
        #inputs = tf.transpose(X, [1,0])
        inputs = tf.reshape(X, [-1, n_steps, n_inputs])
        #inputs = tf.transpose(inputs, [0, 1, 2])
        print(inputs)
        print(inputs)
        #X = tf.reshape(X, [-1, n_steps, n_inputs])
        y_ = tf.placeholder(tf.int64, [None], name='y-input')
        lbls = tf.reshape(y_, [-1, n_steps])
        
      #Dropout probability placeholder
      keep_prob = tf.placeholder(tf.float32, name='keep_prob')
      tf.summary.scalar('keep_probability', keep_prob)
      
      #initial_state = tf.placeholder(tf.float32, [None, None], name='initial_state')
      ###### Graph definition
     
      rnn_out, rnn_state = MultiLSTM_layer(inputs, n_steps, n_neurons, n_lstm_layers, None, keep_prob)
    
      #Output layer, activation identity!!
      y = FC_layer(rnn_out, n_class, 'output_FC', act=tf.identity)
    
      #Optimizer and gradient descent definition
      #Sparse cross entropy as loss function
      with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
          cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=lbls, logits=y, 
                                                                 reduction=tf.losses.Reduction.NONE)
          cross_entropy = tf.reduce_mean(cross_entropy, axis=0)


      tf.summary.scalar('cross_entropy', tf.reduce_mean(cross_entropy))
    
      print(tf.trainable_variables())
      #Optimizer definition (keep ADAM)
      with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    
      #Validation and accuracy computation
      with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
          correct_prediction = tf.equal(tf.argmax(y, 2), lbls)
        with tf.name_scope('accuracy'):
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
          tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.logdir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.logdir + '/test')

  
  sess.run(tf.global_variables_initializer())

  #------Session run, describes the feeding of the graph------
  next_element = iterator.get_next()
  next_validation_element = validation_iterator.get_next()


  def get_next_batch():
    return sess.run(next_element)   

  def get_test_data():
    return sess.run(next_validation_element)

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = get_next_batch()
      k = FLAGS.dropout
    else:
      xs, ys = get_test_data()
      k = 1.0
    #print(xs)  
    #last_state = np.zeros([FLAGS.batch_size, FLAGS.n_neurons*2*FLAGS.n_layers])
    #print("\n\n\n\n\n")
    return {X: xs, y_: ys, keep_prob: k}

  #Loop over epochs
  

  
  i=0
  for epoch in range(n_epochs):    
    sess.run(iterator.initializer, feed_dict={features_ph: features, labels_ph: labels})
    #last_state = np.zeros([batch_size, n_neurons*2*n_layers])
  # Record summaries and test-set accuracy
    while True:
        try:
            i+=1  
            if i % 10 == 0  :  
              sess.run(validation_iterator.initializer, 
                       feed_dict={validation_features_ph: validation_features, 
                                  validation_labels_ph: validation_labels})
              summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
              test_writer.add_summary(summary, i)
              print('Accuracy at step %s, epoch %s: %s' % (i, epoch, acc))              
            # Record train set summaries, and train
            else:   
              # Record execution stats  
              if i % 100 == 99:  
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
                
              # Record a summary
              else:  
                #summary, _ = sess.run([iterator.initializer, merged, train_step], feed_dict=feed_dict(True))
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
        except tf.errors.OutOfRangeError:
            break

  train_writer.close()
  test_writer.close()


#------Main function, overwrites a new log if it already exists------
def main(_):
  if tf.gfile.Exists(FLAGS.logdir):
    tf.gfile.DeleteRecursively(FLAGS.logdir)
  tf.gfile.MakeDirs(FLAGS.logdir)
  with tf.Graph().as_default():
    train()

#Run from terminal with python LSTM.py [--options arg]
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--n_layers', type=int, default=2,
                      help='Number of layers.')
  parser.add_argument('--n_epochs', type=int, default=10,
                      help='Number of epochs to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=1,
                      help='Keep probability for training dropout.')
  parser.add_argument('--n_neurons', type=int, default=150,
                      help='Number of neurons in LSTM cell.')
  parser.add_argument('--n_steps', type=int, default=30,
                      help='Number of RNN steps.')
  parser.add_argument('--batch_size', type=int, default=100,
                      help='Size of the training batch.')
  parser.add_argument('nn_type', type=str, default='dense',
                      help='type [dense] or [lstm]')
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory for storing input data')
  parser.add_argument(
      '--logdir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           './logs'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
