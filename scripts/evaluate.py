import tensorflow as tf
import numpy as np
from sklearn.metrics import average_precision_score
import time, datetime, os, sys, json, pickle
from model import network
from utils import printer


config = json.loads(open('./data/config').read())
FLAGS = tf.app.flags.FLAGS

# overall settings
tf.app.flags.DEFINE_string('model', 'cnn_hier', 'neural models to encode sentences')
tf.app.flags.DEFINE_bool('use_baseline', False, 'baseline or hier')
tf.app.flags.DEFINE_string('mode', 'pone', 'test mode')
tf.app.flags.DEFINE_string('gpu', '2', 'gpu(s) to use')
tf.app.flags.DEFINE_bool('allow_growth', True, 'occupying gpu(s) gradually')
tf.app.flags.DEFINE_string('checkpoint_path','./outputs/ckpt/','path to store model')
tf.app.flags.DEFINE_string('logits_path','./outputs/logits/','path to store model')
tf.app.flags.DEFINE_string('data_path', './data/', 'path to load data')
tf.app.flags.DEFINE_integer('batch_size',262,'instance(entity pair) numbers to use each training(testing) time')

# training settings
tf.app.flags.DEFINE_integer('max_epoch',30,'maximum of training epochs')
tf.app.flags.DEFINE_integer('save_epoch',2,'frequency of training epochs')
tf.app.flags.DEFINE_float('learning_rate',0.2,'entity numbers used each training time')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('keep_prob',1.0,'dropout rate')

# test_settings
tf.app.flags.DEFINE_bool('test_single', True, 'only test one checkpoint')
tf.app.flags.DEFINE_integer('test_start_ckpt',1,'first epoch to test')
tf.app.flags.DEFINE_integer('test_end_ckpt',30,'last epoch to test')
tf.app.flags.DEFINE_float('test_sleep',10,'time units to sleep ')
tf.app.flags.DEFINE_bool('test_use_step',False,'test step instead of epoch')
tf.app.flags.DEFINE_integer('test_start_step',0*1832,'first step to test')
tf.app.flags.DEFINE_integer('test_end_step',30*1832,'last step to test')
tf.app.flags.DEFINE_integer('test_step',1832,'step to add per test')

# parameters
tf.app.flags.DEFINE_integer('word_size', config['word_size'],'maximum of relations')
tf.app.flags.DEFINE_integer('hidden_size',230,'hidden feature size')
tf.app.flags.DEFINE_integer('pos_size',5,'position embedding size')

# statistics
tf.app.flags.DEFINE_integer('max_length',config['fixlen'],'maximum of number of words in one sentence')
tf.app.flags.DEFINE_integer('pos_num', config['maxlen'] * 2 + 1,'number of position embedding vectors')
tf.app.flags.DEFINE_integer('num_classes', len(config['relation2id']),'maximum of relations')
tf.app.flags.DEFINE_integer('vocabulary_size', len(config['word2id']),'maximum of relations')

tf_configs = tf.ConfigProto()
tf_configs.gpu_options.allow_growth = FLAGS.allow_growth

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

def main(_):

    pter = printer.Printer()

    pter('reading test data')

    init_file = FLAGS.data_path+'initial_vectors/' + ('init_vec_pcnn' if FLAGS.model[:4].lower() == 'pcnn' and not FLAGS.use_baseline else 'init_vec')
    init_vec = pickle.load(open(init_file, 'rb'))

    mode = FLAGS.mode
    export_path = FLAGS.data_path

    if mode == 'pr':
        test_instance_triple = np.load(export_path + 'test_entity_pair.npy')
        test_instance_scope = np.load(export_path + 'test_entity_scope.npy')
        test_len = np.load(export_path + 'test_len.npy')
        test_label = np.load(export_path + 'test_label.npy')
        test_word = np.load(export_path + 'test_word.npy')
        test_pos1 = np.load(export_path + 'test_pos1.npy')
        test_pos2 = np.load(export_path + 'test_pos2.npy')
        test_mask = np.load(export_path + 'test_mask.npy')
        exclude_na_flatten_label = np.load(export_path + 'all_true_label.npy')
    else:
        test_instance_triple = np.load(export_path + 'pn/test_entity_pair_pn.npy')
        test_instance_scope = np.load(export_path + 'pn/test_entity_scope_'+mode+'.npy')
        test_len = np.load(export_path + 'pn/test_len_'+mode+'.npy')
        test_label = np.load(export_path + 'pn/test_label_'+mode+'.npy')
        test_word = np.load(export_path + 'pn/test_word_'+mode+'.npy')
        test_pos1 = np.load(export_path + 'pn/test_pos1_'+mode+'.npy')
        test_pos2 = np.load(export_path + 'pn/test_pos2_'+mode+'.npy')
        test_mask = np.load(export_path + 'pn/test_mask_'+mode+'.npy')
        exclude_na_flatten_label = np.load(export_path + 'pn/true_label.npy')

    exclude_na_label = np.reshape(exclude_na_flatten_label, [-1, FLAGS.num_classes-1])
    index_non_zero = np.sum(exclude_na_label, 0) > 0

    print('reading test data finished')

    print('entity pairs     : %d' % (len(test_instance_triple)))
    print('sentences        : %d' % (len(test_len)))
    print('relations        : %d' % (FLAGS.num_classes))
    print('word size        : %d' % (FLAGS.word_size))
    print('position size    : %d' % (FLAGS.pos_size))
    print('hidden size      : %d' % (FLAGS.hidden_size))

    print('building network...')
    sess = tf.Session(config=tf_configs)
    if FLAGS.use_baseline:
        model = network.nre_baseline(is_training=False, init_vec=init_vec)
    else:
        model = network.nre(is_training=False, init_vec=init_vec)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print('building finished...')

    def test_step(word, pos1, pos2, mask, leng, label_index, label, scope):
        feed_dict = {
            model.word: word,
            model.pos1: pos1,
            model.pos2: pos2,
            model.mask: mask,
            model.len : leng,
            model.label_index: label_index,
            model.label: label,
            model.scope: scope,
            model.keep_prob: FLAGS.keep_prob
        }
        output = sess.run(model.test_output, feed_dict)
        return output

    start_ckpt = FLAGS.test_start_ckpt
    start_step = FLAGS.test_start_step
    if FLAGS.test_single:
        end_ckpt = FLAGS.test_start_ckpt
        end_step = FLAGS.test_start_step
    else:
        end_ckpt = FLAGS.test_end_ckpt
        end_step = FLAGS.test_end_step

    if FLAGS.test_use_step == False:
        iteration_list = range(start_ckpt, end_ckpt+1)
    else:
        iteration_list = range(start_step, end_step+1, FLAGS.test_step)

    for iters in iteration_list:

        if FLAGS.test_use_step == False:
            pter('waiting for epoch' + str(FLAGS.save_epoch*iters)+'...')
            while FLAGS.model+"-"+str(FLAGS.save_epoch*1832*iters)+'.index' not in os.listdir(FLAGS.checkpoint_path):
                time.sleep(FLAGS.test_sleep)
            saver.restore(sess, FLAGS.checkpoint_path + FLAGS.model+"-"+str(FLAGS.save_epoch * 1832 * iters))
        else:
            pter('waiting for step' + str(iters)+'...')
            while FLAGS.model+"-"+str(iters)+'.index' not in os.listdir(FLAGS.checkpoint_path):
                time.sleep(FLAGS.test_sleep)
            saver.restore(sess, FLAGS.checkpoint_path + FLAGS.model+"-"+str(iters))

        stack_output = []
        
        iteration = len(test_instance_scope) // FLAGS.batch_size
        for i in range(iteration):
            if FLAGS.test_use_step == False:
                pter('running '+str(i)+'/'+str(iteration)+' for epoch ' + str(FLAGS.save_epoch*iters) + '...')
            else:
                pter('running '+str(i)+'/'+str(iteration)+' for step ' + str(iters) + '...')
            input_scope = test_instance_scope[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size]
            index = []
            scope = [0]
            label = []
            for num in input_scope:
                index = index + list(range(num[0], num[1] + 1))
                label.append(test_label[num[0]])
                scope.append(scope[len(scope)-1] + num[1] - num[0] + 1)
            label_ = np.zeros((FLAGS.batch_size, FLAGS.num_classes))
            label_[np.arange(FLAGS.batch_size), label] = 1
            output = test_step(test_word[index,:], test_pos1[index,:], test_pos2[index,:], 
                test_mask[index,:], test_len[index], test_label[index], label_, np.array(scope))
            stack_output.append(output)

        if FLAGS.test_use_step == False:
            pter('evaluating epoch ' + str(FLAGS.save_epoch*iters) + '...')
        else:
            pter('evaluating step ' + str(iters) + '...')

        stack_output = np.concatenate(stack_output, axis=0)
        exclude_na_output = stack_output[:,1:]
        exclude_na_flatten_output = np.reshape(stack_output[:,1:],(-1))

        if mode == 'pr':
            m = average_precision_score(exclude_na_flatten_label, exclude_na_flatten_output)
            M = average_precision_score(exclude_na_label[:,index_non_zero], exclude_na_output[:,index_non_zero], average='macro')
            np.save(FLAGS.logits_path+FLAGS.model+str(iters), exclude_na_flatten_output)
            print(m, M)
        else:
            order = np.argsort(-exclude_na_flatten_output)
            print(np.mean(exclude_na_flatten_label[order[:100]]))
            print(np.mean(exclude_na_flatten_label[order[:200]]))
            print(np.mean(exclude_na_flatten_label[order[:300]]))

if __name__ == "__main__":
    tf.app.run()
