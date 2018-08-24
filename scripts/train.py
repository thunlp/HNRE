import tensorflow as tf
import numpy as np
from sklearn.metrics import average_precision_score
import time, datetime, os, sys, json, pickle
from model import network


config = json.loads(open("./data/config",'r').read())

FLAGS = tf.app.flags.FLAGS

# overall
tf.app.flags.DEFINE_string('model', 'cnn', 'neural models to encode sentences')
tf.app.flags.DEFINE_bool('use_baseline', False, 'baseline or hier')
tf.app.flags.DEFINE_string('gpu', '3', 'gpu to use')
tf.app.flags.DEFINE_bool('allow_growth', False, 'memory growth')
tf.app.flags.DEFINE_string('data_path', './data/', 'path to load data')
tf.app.flags.DEFINE_string('model_dir','./output/ckpt/','path to store model')
tf.app.flags.DEFINE_string('summary_dir','./outputs/summary','path to store summary_dir')
tf.app.flags.DEFINE_integer('batch_size',160,'entity numbers used each training time')
# training
tf.app.flags.DEFINE_integer('max_epoch',40,'maximum of training epochs')
tf.app.flags.DEFINE_integer('save_epoch',2,'frequency of training epochs')
tf.app.flags.DEFINE_integer('restore_epoch',0,'epoch to continue training')
tf.app.flags.DEFINE_float('learning_rate',0.2,'learning rate')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('keep_prob',0.5,'dropout rate')
# parameters
tf.app.flags.DEFINE_integer('word_size', config['word_size'],'maximum of relations')
tf.app.flags.DEFINE_integer('hidden_size',230,'hidden feature size')
tf.app.flags.DEFINE_integer('pos_size',5,'position embedding size')
# statistics
tf.app.flags.DEFINE_integer('max_length',config['fixlen'],'maximum of number of words in one sentence')
tf.app.flags.DEFINE_integer('pos_num', config['maxlen'] * 2 + 1,'number of position embedding vectors')
tf.app.flags.DEFINE_integer('num_classes', len(config['relation2id']),'maximum of relations')
tf.app.flags.DEFINE_integer('vocabulary_size', len(config['word2id']),'maximum of relations')


os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
tf_configs = tf.ConfigProto()
tf_configs.gpu_options.allow_growth = FLAGS.allow_growth

export_path = FLAGS.data_path

def main(_):

    print('reading training data')

    init_file = './data/initial_vectors/' + ('init_vec_pcnn' if FLAGS.model[:4].lower() == 'pcnn' and not FLAGS.use_baseline else 'init_vec')
    init_vec = pickle.load(open(init_file, 'rb'))

    instance_triple = np.load(export_path + 'train_instance_triple.npy')
    instance_scope = np.load(export_path + 'train_instance_scope.npy')
    train_len = np.load(export_path + 'train_len.npy')
    train_label = np.load(export_path + 'train_label.npy')
    train_word = np.load(export_path + 'train_word.npy')
    train_pos1 = np.load(export_path + 'train_pos1.npy')
    train_pos2 = np.load(export_path + 'train_pos2.npy')
    train_mask = np.load(export_path + 'train_mask.npy')

    print('reading finished')

    print('instances         : %d' % (len(instance_triple)))
    print('sentences         : %d' % (len(train_len)))
    print('relations         : %d' % (FLAGS.num_classes))
    print('word size         : %d' % (FLAGS.word_size))
    print('position size     : %d' % (FLAGS.pos_size))
    print('hidden size       : %d' % (FLAGS.hidden_size))

    print('building network...')
    sess = tf.Session(config=tf_configs)
    if FLAGS.use_baseline:
        model = network.nre_baseline(is_training=True, init_vec=init_vec)
    else:
        model = network.nre(is_training=True, init_vec=init_vec)
    global_step = tf.Variable(0,name='global_step',trainable=False)
    learning_rate = FLAGS.learning_rate
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(model.loss, global_step = global_step)
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=20)
    print('building finished')

    def train_step(word, pos1, pos2, mask, leng, label_index, label, scope):
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
        _, step, loss, summary, output, correct_predictions = sess.run([train_op, global_step, model.loss, merged_summary, model.output, model.correct_predictions], feed_dict)
        summary_writer.add_summary(summary, step)
        return output, loss, correct_predictions


    train_order = list(range(len(instance_triple)))

    save_epoch = FLAGS.save_epoch

    if FLAGS.restore_epoch > 0:
        saver.restore(sess, FLAGS.model_dir + FLAGS.model+"-"+str(1832*FLAGS.restore_epoch))
        print('restored model from epoch {}'.format(FLAGS.restore_epoch))

    for one_epoch in range(FLAGS.max_epoch):

        np.random.shuffle(train_order)
        s1 = 0.0
        s2 = 0.0
        tot1 = 0.0
        tot2 = 0.0
        loss_sum = 0.0
        step_sum = 0.0
        for i in range(int(len(train_order)/float(FLAGS.batch_size))):
            input_scope = np.take(instance_scope, train_order[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size], axis=0)
            index = []
            scope = [0]
            label = []
            for num in input_scope:
                index = index + list(range(num[0], num[1] + 1))
                label.append(train_label[num[0]])
                scope.append(scope[len(scope)-1] + num[1] - num[0] + 1)
            label_ = np.zeros((FLAGS.batch_size, FLAGS.num_classes))
            label_[np.arange(FLAGS.batch_size), label] = 1
            output, loss, correct_predictions = train_step(train_word[index,:], train_pos1[index,:], train_pos2[index,:], 
                train_mask[index,:], train_len[index],train_label[index], label_, np.array(scope))
            num = 0
            s = 0
            for num in correct_predictions:
                if label[s] == 0:
                    tot1 += 1.0
                    if num:
                        s1+= 1.0
                else:
                    tot2 += 1.0
                    if num:
                        s2 += 1.0
                s = s + 1
            loss_sum += loss
            step_sum += 1.0
            if tot1 == 0:
                tot1 += 1
            if tot2 == 0:
                tot2 += 1

            time_str = datetime.datetime.now().isoformat().replace('T',' ')
            temp_str = 'epoch {0:0>3}/{1:0>3} step {2:0>4} time {3:26} | loss : {4:1.8f} | NA accuracy: {5:1.6f} | not NA accuracy: {6:1.6f}\r'.format(FLAGS.restore_epoch + one_epoch + 1, FLAGS.restore_epoch + FLAGS.max_epoch, i, time_str, loss, s1 / tot1, s2 / tot2)
            sys.stdout.write(temp_str)
            sys.stdout.flush()
            
        temp_str = 'epoch {0:0>3}/{1:0>3} step {2:0>4} time {3:26} | loss : {4:1.8f} | NA accuracy: {5:1.6f} | not NA accuracy: {6:1.6f}\r'.format(FLAGS.restore_epoch + one_epoch + 1, FLAGS.restore_epoch + FLAGS.max_epoch, i, time_str, loss_sum / step_sum, s1 / tot1, s2 / tot2)
        print(temp_str)
        current_step = tf.train.global_step(sess, global_step)
        if (one_epoch + 1) % save_epoch == 0:
            sys.stdout.write('saving model...' + '\r')
            sys.stdout.flush()
            path = saver.save(sess,FLAGS.model_dir + FLAGS.model, global_step=current_step)
            sys.stdout.write('have saved model to ' + path + '\r')
            sys.stdout.flush()

if __name__ == "__main__":
    tf.app.run()
