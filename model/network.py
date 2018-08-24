import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import xavier_initializer as xavier

FLAGS = tf.app.flags.FLAGS

class NN(object):

    def __init__(self, is_training, init_vec):

        self.word = tf.placeholder(dtype=tf.int32,shape=[None, FLAGS.max_length], name='input_word')
        self.pos1 = tf.placeholder(dtype=tf.int32,shape=[None, FLAGS.max_length], name='input_pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32,shape=[None, FLAGS.max_length], name='input_pos2')
        self.mask = tf.placeholder(dtype=tf.int32,shape=[None, FLAGS.max_length],name='input_mask')
        self.len = tf.placeholder(dtype=tf.int32,shape=[None],name='input_len')
        self.label_index = tf.placeholder(dtype=tf.int32,shape=[None], name='label_index')
        self.label = tf.placeholder(dtype=tf.float32,shape=[FLAGS.batch_size, FLAGS.num_classes], name='input_label')
        self.scope = tf.placeholder(dtype=tf.int32,shape=[FLAGS.batch_size+1], name='scope')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.hier = init_vec['relation_levels'].shape[1]
        self.relation_levels = tf.constant(init_vec['relation_levels'], shape=[FLAGS.num_classes, self.hier], 
            dtype=tf.int32, name='relation_levels')
        self.layer = (1 + np.max(init_vec['relation_levels'], 0)).astype(np.int32)

        word_size = FLAGS.word_size
        vocab_size = FLAGS.vocabulary_size - 2

        with tf.variable_scope("embedding-lookup", initializer=xavier(), dtype=tf.float32):

            temp_word_embedding = self._GetVar(init_vec=init_vec, key='wordvec', name='temp_word_embedding', 
                shape=[vocab_size, word_size],trainable=True)
            unk_word_embedding = self._GetVar(init_vec=init_vec, key='unkvec', name='unk_embedding', 
                shape=[word_size], trainable=True)
            word_embedding = tf.concat([temp_word_embedding, tf.reshape(unk_word_embedding,[1,word_size]),
                tf.reshape(tf.constant(np.zeros(word_size),dtype=tf.float32),[1,word_size])],0)

            temp_pos1_embedding = self._GetVar(init_vec=init_vec, key='pos1vec', name='temp_pos1_embedding',
                shape=[FLAGS.pos_num,FLAGS.pos_size], trainable=True)
            temp_pos2_embedding = self._GetVar(init_vec=init_vec, key='pos2vec', name='temp_pos2_embedding',
                shape=[FLAGS.pos_num,FLAGS.pos_size], trainable=True)
            pos1_embedding = tf.concat([temp_pos1_embedding,
                tf.reshape(tf.constant(np.zeros(FLAGS.pos_size,dtype=np.float32)),[1, FLAGS.pos_size])],0)
            pos2_embedding = tf.concat([temp_pos2_embedding,
                tf.reshape(tf.constant(np.zeros(FLAGS.pos_size,dtype=np.float32)),[1, FLAGS.pos_size])],0)

            input_word = tf.nn.embedding_lookup(word_embedding, self.word)
            input_pos1 = tf.nn.embedding_lookup(pos1_embedding, self.pos1)
            input_pos2 = tf.nn.embedding_lookup(pos2_embedding, self.pos2)
            self.input_embedding = tf.concat(values = [input_word, input_pos1, input_pos2], axis = 2)

        self.hidden_size, self.sentence_encoder = self._GetEncoder(FLAGS.model, is_training)

    def _GetVar(self, init_vec, key, name, shape=None, initializer=None, trainable=True):
        
        if init_vec is not None and key in init_vec:
            print('using pretrained {} and is {}'.format(key, 'trainable' if trainable else 'not trainable'))
            return tf.get_variable(name = name, initializer = init_vec[key], trainable = trainable)
        else:
            return tf.get_variable(name = name, shape = shape, initializer = initializer, trainable = trainable)

    def _GetEncoder(self, model, is_training):

        if model.lower()[:3] == 'cnn':
            return FLAGS.hidden_size, self.EncoderCNN
        elif model.lower()[:4] == 'pcnn':
            return FLAGS.hidden_size * 3, self.EncoderPCNN
        elif model.lower()[:4] == 'lstm':
            return FLAGS.hidden_size * 2, self.EncoderLSTM
        else:
            raise Exception
        
    def EncoderCNN(self, is_training, init_vec=None):

        with tf.variable_scope("sentence-encoder", dtype=tf.float32, initializer=xavier(), reuse=tf.AUTO_REUSE):
            input_dim = self.input_embedding.shape[2]
            input_sentence = tf.expand_dims(self.input_embedding, axis=1)
            with tf.variable_scope("conv2d"):
                conv_kernel = self._GetVar(init_vec=init_vec,key='convkernel',name='kernel',
                    shape=[1,3,input_dim,FLAGS.hidden_size],trainable=True)
                conv_bias = self._GetVar(init_vec=init_vec,key='convbias',name='bias',shape=[FLAGS.hidden_size],trainable=True)
            x = tf.layers.conv2d(inputs = input_sentence, filters=FLAGS.hidden_size, 
                kernel_size=[1,3], strides=[1, 1], padding='same', reuse=tf.AUTO_REUSE)
            x = tf.reduce_max(x, axis=2)
            x = tf.nn.relu(tf.squeeze(x, 1))

        return x

    def EncoderPCNN(self, is_training, init_vec=None):
        
        with tf.variable_scope("sentence-encoder", dtype=tf.float32, initializer=xavier(), reuse=tf.AUTO_REUSE):
            input_dim = self.input_embedding.shape[2]
            mask_embedding = tf.constant([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
            pcnn_mask = tf.nn.embedding_lookup(mask_embedding, self.mask)
            input_sentence = tf.expand_dims(self.input_embedding, axis=1)
            with tf.variable_scope("conv2d"):
                conv_kernel = self._GetVar(init_vec=init_vec,key='convkernel',name='kernel',
                    shape=[1,3,input_dim,FLAGS.hidden_size],trainable=True)
                conv_bias = self._GetVar(init_vec=init_vec,key='convbias',name='bias',shape=[FLAGS.hidden_size],trainable=True)
            x = tf.layers.conv2d(inputs = input_sentence, filters=FLAGS.hidden_size, 
                kernel_size=[1,3], strides=[1, 1], padding='same', reuse=tf.AUTO_REUSE)
            x = tf.reshape(x, [-1, FLAGS.max_length, FLAGS.hidden_size, 1])
            x = tf.reduce_max(tf.reshape(pcnn_mask, [-1, 1, FLAGS.max_length, 3]) * tf.transpose(x,[0, 2, 1, 3]), axis = 2)
            x = tf.nn.relu(tf.reshape(x, [-1, FLAGS.hidden_size * 3]))

        return x

    def EncoderLSTM(self, is_training, init_vec=None):

        with tf.variable_scope("sentence-encoder", dtype=tf.float32, initializer=xavier(), reuse=tf.AUTO_REUSE):
            input_sentence = tf.layers.dropout(self.input_embedding, rate = self.keep_prob, training = is_training)
            fw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_size, state_is_tuple=True)
            bw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_size, state_is_tuple=True)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                            fw_cell, bw_cell, input_sentence,
                            sequence_length = self.len,
                            dtype = tf.float32,
                            scope = 'bi-dynamic-rnn')
            fw_states, bw_states = states
            if isinstance(fw_states, tuple):
                fw_states = fw_states[0]
                bw_states = bw_states[0]
            x = tf.concat(states, axis=1)
            
        return x

class nre(NN):

    def __init__(self, is_training, init_vec):

        NN.__init__(self, is_training, init_vec)

        x = self.sentence_encoder(is_training, init_vec)

        with tf.variable_scope("sentence-level-attention", initializer=xavier(), dtype=tf.float32):

            relation_matrixs = []

            for i in range(self.hier):
                relation_matrixs.append(self._GetVar(init_vec=init_vec, key='relmat'+str(i),
                    name='relation_matrix_l'+str(i), shape=[self.layer[i], self.hidden_size]))

            label_layer = tf.nn.embedding_lookup(self.relation_levels, self.label_index)
            attention_logits = []
            for i in range(self.hier):
                current_relation = tf.nn.embedding_lookup(relation_matrixs[i], label_layer[:, i])
                attention_logits.append(tf.reduce_sum(current_relation * x, 1))

            attention_logits_stack = tf.stack(attention_logits)
            attention_score_hidden = tf.concat([
                tf.nn.softmax(attention_logits_stack[:,self.scope[i]:self.scope[i+1]]) for i in range(FLAGS.batch_size)], 1)

            tower_repre = []
            for i in range(FLAGS.batch_size):
                sen_matrix = x[self.scope[i]:self.scope[i+1]]
                layer_score = attention_score_hidden[:,self.scope[i]:self.scope[i+1]]
                layer_repre = tf.reshape(layer_score @ sen_matrix, [-1])
                tower_repre.append(layer_repre)

            stack_repre = tf.layers.dropout(tf.stack(tower_repre), rate = 1 - self.keep_prob, training = is_training)

        with tf.variable_scope("loss", dtype=tf.float32, initializer=xavier()):

            discrimitive_matrix = self._GetVar(init_vec=init_vec, key='disckernel',
                name='discrimitive_matrix', shape=[FLAGS.num_classes, self.hidden_size * self.hier])

            bias = self._GetVar(init_vec=init_vec, key='discbias',
                name='bias', shape=[FLAGS.num_classes], initializer=tf.zeros_initializer())

            logits = tf.matmul(stack_repre, discrimitive_matrix, transpose_b=True) + bias
            regularizer = tf.contrib.layers.l2_regularizer(FLAGS.weight_decay)
            l2_loss = tf.contrib.layers.apply_regularization(regularizer=regularizer, weights_list=tf.trainable_variables())
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=logits)) + l2_loss 

            self.output = tf.nn.softmax(logits)
            tf.summary.scalar('loss',self.loss)
            self.predictions = tf.argmax(logits, 1, name="predictions")
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

        if not is_training:

            with tf.variable_scope("test"):

                test_attention_scores = []
                for i in range(self.hier):
                    current_relation = tf.nn.embedding_lookup(relation_matrixs[i], self.relation_levels[:, i])
                    current_logit = tf.matmul(current_relation, x, transpose_b=True)
                    current_score = tf.concat([
                        tf.nn.softmax(current_logit[:,self.scope[j]:self.scope[j+1]]) for j in range(FLAGS.batch_size)], 1)
                    test_attention_scores.append(current_score)
                test_attention_scores_stack = tf.stack(test_attention_scores, 1)

                test_tower_output = []
                for i in range(FLAGS.batch_size):
                    test_sen_matrix = tf.tile(tf.expand_dims(x[self.scope[i]:self.scope[i+1]], 0), [FLAGS.num_classes, 1, 1])
                    test_layer_score = test_attention_scores_stack[:,:,self.scope[i]:self.scope[i+1]]
                    test_layer_repre = tf.reshape(test_layer_score @ test_sen_matrix, [FLAGS.num_classes, -1])
                    test_logits = tf.matmul(test_layer_repre, discrimitive_matrix, transpose_b=True) + bias
                    test_output = tf.diag_part(tf.nn.softmax(test_logits))
                    test_tower_output.append(test_output)

                test_stack_output = tf.reshape(tf.stack(test_tower_output),[FLAGS.batch_size, FLAGS.num_classes])
                self.test_output = test_stack_output




class nre_baseline(NN):

    def __init__(self, is_training, init_vec = None):

        NN.__init__(self, is_training, init_vec)

        x = self.sentence_encoder(is_training, init_vec)

    
        with tf.variable_scope("sentence-level-attention", initializer=xavier(),dtype=tf.float32):

            relation_matrix = self._GetVar(init_vec=init_vec, key='relmat',
                name='relation_matrix', shape=[FLAGS.num_classes, self.hidden_size])

            current_relation = tf.nn.embedding_lookup(relation_matrix, self.label_index)
            attention_logit = tf.reduce_sum(x * current_relation, 1)

            tower_repre = []
            for i in range(FLAGS.batch_size):
                sen_matrix = x[self.scope[i]:self.scope[i+1]]
                attention_score = tf.nn.softmax(tf.reshape(attention_logit[self.scope[i]:self.scope[i+1]], [1, -1]))
                final_repre = tf.reshape(tf.matmul(attention_score, sen_matrix), [self.hidden_size])
                tower_repre.append(final_repre)
            stack_repre = tf.layers.dropout(tf.stack(tower_repre), rate = 1 - self.keep_prob, training = is_training)

        with tf.variable_scope("loss",dtype=tf.float32,initializer=xavier()):

            discrimitive_matrix = self._GetVar(init_vec=init_vec, key='discmat',
                name='discrimitive_matrix', shape=[FLAGS.num_classes, self.hidden_size])

            bias = self._GetVar(init_vec=init_vec, key='disc_bias',
                name='bias', shape=[FLAGS.num_classes], initializer=tf.zeros_initializer())

            logits = tf.matmul(stack_repre, discrimitive_matrix, transpose_b=True) + bias
            self.output = tf.nn.softmax(logits)

            regularizer = tf.contrib.layers.l2_regularizer(FLAGS.weight_decay)
            l2_loss = tf.contrib.layers.apply_regularization(regularizer=regularizer, weights_list=tf.trainable_variables())
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=logits)) + l2_loss 

            tf.summary.scalar('loss',self.loss)
            self.predictions = tf.argmax(logits, 1, name="predictions")
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

        if not is_training:

            with tf.variable_scope("test"):
                test_attention_logit = tf.matmul(x, relation_matrix, transpose_b=True)
                test_tower_output = []
                for i in range(FLAGS.batch_size):
                    test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[self.scope[i]:self.scope[i+1],:]))
                    test_final_repre = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i+1]])
                    test_logits = tf.matmul(test_final_repre, discrimitive_matrix, transpose_b=True) + bias * 3
                    test_output = tf.diag_part(tf.nn.softmax(test_logits))
                    test_tower_output.append(test_output)
                test_stack_output = tf.reshape(tf.stack(test_tower_output),[FLAGS.batch_size, FLAGS.num_classes])
                self.test_output = test_stack_output
