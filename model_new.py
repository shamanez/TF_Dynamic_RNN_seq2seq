# Working with TF commit 24466c2e6d32621cd85f0a78d47df6eed2c5c5a6
#this was originally taken from the ematvey tutorial on tensorflow in Github. Special thanks to him 
#Link to his tutorial - https://github.com/ematvey/tensorflow-seq2seq-tutorials
#I commented the functions  

import math

import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.layers import safe_embedding_lookup_sparse as embedding_lookup_unique
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell

import helpers


class Seq2SeqModel():
    """Seq2Seq model usign blocks from new `tf.contrib.seq2seq`.
    Requires TF 1.0.0-alpha"""

    PAD = 0
    EOS = 1

    def __init__(self, encoder_cell, decoder_cell, vocab_size, embedding_size,
                 bidirectional=True,
                 attention=False,
                 debug=False):
        self.debug = debug
        self.bidirectional = bidirectional
        self.attention = attention
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell

        self._make_graph()

    @property
    def decoder_hidden_units(self):
        # @TODO: is this correct for LSTMStateTuple?
        return self.decoder_cell.output_size

    def _make_graph(self):
        if self.debug:
            self._init_debug_inputs()
        else:
            self._init_placeholders()

        self._init_decoder_train_connectors()
        self._init_embeddings()

        if self.bidirectional:
            self._init_bidirectional_encoder()
        else:
            self._init_simple_encoder()

        self._init_decoder()

        self._init_optimizer()

    def _init_debug_inputs(self):
        """ Everything is time-major """
        x = [[5, 6, 7],
             [7, 6, 0],
             [0, 7, 0]]
        xl = [2, 3, 1]
        self.encoder_inputs = tf.constant(x, dtype=tf.int32, name='encoder_inputs')
        self.encoder_inputs_length = tf.constant(xl, dtype=tf.int32, name='encoder_inputs_length')

        self.decoder_targets = tf.constant(x, dtype=tf.int32, name='decoder_targets')
        self.decoder_targets_length = tf.constant(xl, dtype=tf.int32, name='decoder_targets_length')

    def _init_placeholders(self):
        """ Everything is time-major """
        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs',
        )
        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length',
        )

        # required for training, not required for testing
        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets'
        )
        self.decoder_targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_targets_length',
        )

    def _init_decoder_train_connectors(self):
        """
        During training, `decoder_targets`
        and decoder logits. This means that their shapes should be compatible.

        Here we do a bit of plumbing to set this up.
        """
        with tf.name_scope('DecoderTrainFeeds'):
            sequence_size, batch_size = tf.unstack(tf.shape(self.decoder_targets))

            EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS
            PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD

            self.decoder_train_inputs = tf.concat([EOS_SLICE, self.decoder_targets], axis=0)
            self.decoder_train_length = self.decoder_targets_length + 1
           
            decoder_train_targets = tf.concat([self.decoder_targets, PAD_SLICE], axis=0)
            decoder_train_targets_seq_len, _ = tf.unstack(tf.shape(decoder_train_targets))
            decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1,
                                                        decoder_train_targets_seq_len,
                                                        on_value=self.EOS, off_value=self.PAD,
                                                        dtype=tf.int32)
            decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask, [1, 0])
            # hacky way using one_hot to put EOS symbol at the end of target sequence
            decoder_train_targets = tf.add(decoder_train_targets,
                                           decoder_train_targets_eos_mask)

            self.decoder_train_targets = decoder_train_targets

            self.loss_weights = tf.ones([
                batch_size,
                tf.reduce_max(self.decoder_train_length)
            ], dtype=tf.float32, name="loss_weights")

    def _init_embeddings(self):
        with tf.variable_scope("embedding") as scope:

            # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix",
                shape=[self.vocab_size, self.embedding_size],
                initializer=initializer,
                dtype=tf.float32)

            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.encoder_inputs)

            self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.decoder_train_inputs)

    def _init_simple_encoder(self):
        with tf.variable_scope("Encoder") as scope:
            (self.encoder_outputs, self.encoder_state) = (
                tf.nn.dynamic_rnn(cell=self.encoder_cell,
                                  inputs=self.encoder_inputs_embedded,
                                  sequence_length=self.encoder_inputs_length,
                                  time_major=True,
                                  dtype=tf.float32)
                )

    def _init_bidirectional_encoder(self):
     
        with tf.variable_scope("BidirectionalEncoder") as scope:

            ((encoder_fw_outputs,  #hidden cell outputs  in forward pass
              encoder_bw_outputs), #hidden cell outputs in backword pass
             (encoder_fw_state,       #final cell state fwrd                                              #setting up the enoceder 
              encoder_bw_state)) = (   #final cell state backward
                tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell,
                                                cell_bw=self.encoder_cell,
                                                inputs=self.encoder_inputs_embedded,
                                                sequence_length=self.encoder_inputs_length,
                                                time_major=True,
                                                dtype=tf.float32)
                )

            self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)  #enoceder outputs hidden states of each time atep
              
            if isinstance(encoder_fw_state, LSTMStateTuple):
                print("I am inside")
                encoder_state_c = tf.concat(
                    (encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')   #cell state 
                encoder_state_h = tf.concat(
                    (encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')     #final output state
                self.encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

            elif isinstance(encoder_fw_state, tf.Tensor):        
                self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state), 1, name='bidirectional_concat')
                print( "sssssssssssssssssssssssssssssssssssssssssss",self.encoder_state)

    def _init_decoder(self):
        with tf.variable_scope("Decoder") as scope:
            def output_fn(outputs):
                return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)  #this is for calculatng outputs. In a greedy way

            if not self.attention:
                decoder_fn_train = seq2seq.simple_decoder_fn_train(encoder_state=self.encoder_state)  #This is the training  function that we used in training  dynamic_rnn_decoder 

#refer to https://github.com/tensorflow/tensorflow/blob/r1.0/tensorflow/contrib/seq2seq/python/ops/decoder_fn.py#L182


                decoder_fn_inference = seq2seq.simple_decoder_fn_inference(  #nference function for a sequence-to-sequence model. It should be used when dynamic_rnn_decoder is in the inference mode.final mode
                    output_fn=output_fn,                           #this returns a decoder function . This function in used inside the dynamicRNN function
                    encoder_state=self.encoder_state,
                    embeddings=self.embedding_matrix,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                    num_decoder_symbols=self.vocab_size,
                )
            else:

                # attention_states: size [batch_size, max_time, num_units]
                attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])    #take the attention status as the encorder hidden states 
      
                (attention_keys,  #Each Encoder hidden status multiplied in fully conected way and list of size [num units*Max_time] 
                attention_values,  #this is attention encoder states 
                attention_score_fn,       #score function of the attention Different ways to compute attention scores  If we input the decoder state , encoder hidden states  this will out put the context vector 
                attention_construct_fn) = seq2seq.prepare_attention(  #this contruct will Function to compute attention vectors. This will output the concatanaded context vector and the attention wuary then make it as a inpit 
                    attention_states=attention_states,
                    attention_option="bahdanau",
                    num_units=self.decoder_hidden_units,
                )
                print("Prininting the number of units .......................")
                print(self.decoder_hidden_units)
                print("Printing the shape of the attetniton values ......................**********************************************")
                print(attention_keys)
                print("Printing the attention score function++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print(attention_score_fn)

#this function can basically initialize input state of the decoder the nthe attention and other stuff then this will be passed to dy_decorder
#decorder_function train will take time, cell_state, cell_input, cell_output, context_state
                decoder_fn_train = seq2seq.attention_decoder_fn_train(  #this is for training the dynamic decorder. This will take care of 
                    encoder_state=self.encoder_state, # final state. We take the biderection and concatanate it (c or h)
                    attention_keys=attention_keys, # The transformation of each encoder outputs 
                    attention_values=attention_values,  #attention encododr status 
                    attention_score_fn=attention_score_fn,   #this will give a context vector
                    attention_construct_fn=attention_construct_fn, #calculating above thinhs  also output the hidden state 
                    name='attention_decoder'
                )
#What can we achieve by running decorder_fn_ ?  done, next state, next input, emit output, next context state
#here the emit_output or cell_output will give the output of cell after all atention - non lieanrity applied 

#this also give the hidden vector output which was concatanated with rnn output and attention vector . Actually concatanated goes throug a linear unit
#next_input = array_ops.concat([cell_input, attention], 1)  #next cell input 
#context_state - this will modify when using the beam search 
#what is the contect state in decorder_fn inside the return funfction of the decorder fn train 
#the following function is same as the above but the only difference is it's use this in the inference .This has a greedy output



#in the inference model cell_output = output_fn(cell_output) . Which means we get logits 
#next_input = array_ops.concat([cell_input, attention], 1)

                decoder_fn_inference = seq2seq.attention_decoder_fn_inference(    #this is used in the inference model 
                    output_fn=output_fn,   #this will predict the output and the narcmax after that attention will be concatenaded 
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,  #doing same 
                    attention_construct_fn=attention_construct_fn,
                    embeddings=self.embedding_matrix,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                    num_decoder_symbols=self.vocab_size,
                )

#following function is to do all the decodinf with the helop of above functions 
#this can use in traning or inferense . But we need two separate finctions for trainin and iference 

#What is this context_state_train : one way to diversify the inference output is to use a stochastic decoder_fn, in which case one would want to store the  decoded outputs, not just the RNN outputs. This can be done by maintaining a TensorArray in context_state and storing the decoded output of each iteration therein



            (self.decoder_outputs_train,  #outputs from the eacah cell [batch_size, max_time, cell.output_size]
             self.decoder_state_train,   #The final state and will be shaped [batch_size, cell.state_size]
             self.decoder_context_state_train) = (  #described above 
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_train, #decoder_fn allows modeling of early stopping, output, state, and next input and context.
                    inputs=self.decoder_train_inputs_embedded,  #inputs to the decoder in the training #in the raning time  only 
                    sequence_length=self.decoder_train_length,#sequence_length is needed at training time, i.e., when inputs is not None, for dynamic unrolling. At test time, when inputs is None, sequence_length is not needed.
                    time_major=True, #input and output shape should be in [max_time, batch_size, ...]
                    scope=scope,
                )
            )

            self.decoder_logits_train = output_fn(self.decoder_outputs_train)  #take the final output hidden status and run them throgh linearl layer #get the argmax 
            self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

            scope.reuse_variables()

            (self.decoder_logits_inference,           #same as above but no input provided. This will take the predicted things as inputs
             self.decoder_state_inference,
             self.decoder_context_state_inference) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_inference,  #difference decorder fucntion 
                    time_major=True,
                    scope=scope,
                )
            )
            self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1, name='decoder_prediction_inference') #predicted output at the each time step

    def _init_optimizer(self):
        logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])  #we input the logits These are the output from the dynmic 
        targets = tf.transpose(self.decoder_train_targets, [1, 0])  #the targets 
        self.loss = seq2seq.sequence_loss(logits=logits, targets=targets, # """Weighted cross-entropy loss for a sequence of logits (per example).
                                          weights=self.loss_weights)       
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss) #onpmize it 
 
    def make_train_inputs(self, input_seq, target_seq):
        inputs_, inputs_length_ = helpers.batch(input_seq) 
        targets_, targets_length_ = helpers.batch(target_seq)
        return {
            self.encoder_inputs: inputs_,
            self.encoder_inputs_length: inputs_length_,
            self.decoder_targets: targets_,
            self.decoder_targets_length: targets_length_,
        }

    def make_inference_inputs(self, input_seq):
        inputs_, inputs_length_ = helpers.batch(input_seq)
        return {
            self.encoder_inputs: inputs_,
            self.encoder_inputs_length: inputs_length_,
        }


def make_seq2seq_model(**kwargs):
    args = dict(encoder_cell=LSTMCell(10),
                decoder_cell=LSTMCell(20),
                vocab_size=10,
                embedding_size=10,
                attention=True,
                bidirectional=True,
                debug=False)
    args.update(kwargs)
    return Seq2SeqModel(**args)


def train_on_copy_task(session, model,
                       length_from=3, length_to=8,
                       vocab_lower=2, vocab_upper=10,
                       batch_size=100,
                       max_batches=5000,
                       batches_in_epoch=1000,
                       verbose=True):

    batches = helpers.random_sequences(length_from=length_from, length_to=length_to,            #generating batches 
                                       vocab_lower=vocab_lower, vocab_upper=vocab_upper,
                                       batch_size=batch_size)
    loss_track = []
    try:
        for batch in range(max_batches+1):
            batch_data = next(batches)
            fd = model.make_train_inputs(batch_data, batch_data)  #this give the data dictionary 
            _, l = session.run([model.train_op, model.loss], fd)
            loss_track.append(l)

            if verbose:
                if batch == 0 or batch % batches_in_epoch == 0:
                    print('batch {}'.format(batch))
                    print('  minibatch loss: {}'.format(session.run(model.loss, fd)))
                    for i, (e_in, dt_pred) in enumerate(zip(
                            fd[model.encoder_inputs].T,
                            session.run(model.decoder_prediction_train, fd).T
                        )):
                        print('  sample {}:'.format(i + 1))
                        print('    enc input           > {}'.format(e_in))
                        print('    dec train predicted > {}'.format(dt_pred))
                        if i >= 2:
                            break
                    print()
    except KeyboardInterrupt:
        print('training interrupted')

    return loss_track


if __name__ == '__main__':
    import sys

    if 'fw-debug' in sys.argv:
        tf.reset_default_graph()
        with tf.Session() as session:
            model = make_seq2seq_model(debug=True)
            session.run(tf.global_variables_initializer())
            session.run(model.decoder_prediction_train)
            session.run(model.decoder_prediction_train)

    elif 'fw-inf' in sys.argv:
        tf.reset_default_graph()
        with tf.Session() as session:
            model = make_seq2seq_model()
            session.run(tf.global_variables_initializer())
            fd = model.make_inference_inputs([[5, 4, 6, 7], [6, 6]])
            inf_out = session.run(model.decoder_prediction_inference, fd)
            print(inf_out)

    elif 'train' in sys.argv:
        tracks = {}

        tf.reset_default_graph()

        with tf.Session() as session:
            model = make_seq2seq_model(attention=True)
            session.run(tf.global_variables_initializer())
            loss_track_attention = train_on_copy_task(session, model)

        tf.reset_default_graph()

        with tf.Session() as session:
            model = make_seq2seq_model(attention=False)
            session.run(tf.global_variables_initializer())
            loss_track_no_attention = train_on_copy_task(session, model)

        import matplotlib.pyplot as plt
        plt.plot(loss_track)
        print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))

    else:
        tf.reset_default_graph()
        session = tf.InteractiveSession()
        model = make_seq2seq_model(debug=False)
        session.run(tf.global_variables_initializer())

        fd = model.make_inference_inputs([[5, 4, 6, 7], [6, 6]])

        inf_out = session.run(model.decoder_prediction_inference, fd)
