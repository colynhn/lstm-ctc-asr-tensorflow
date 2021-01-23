import tensorflow as tf
from tqdm import tqdm
import numpy as np
import time
import scipy.io.wavfile as wav
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

from ctc_train3 import CTC


ctc = CTC(800,128,2,16,0.0001)

#直接调用里边的函数

print("--"*10 + "test begin" + "--"*10 )
print("loading test data...")
test_wav_scp_file = "./scp/test_wav_scp_file"
test_txt_scp_file = "./scp/test_txt_scp_file"
lexicon_file = "./thchs30/data_thchs30/lm_phone/lexicon.txt"
print("loading source data done.")

test_dic = ctc.data_handle(test_wav_scp_file, test_txt_scp_file, lexicon_file)
inputs_test = test_dic["inputs"]
test_sum_num = len(inputs_test)
test_num_batches_per_epoch = int(test_sum_num/ctc._batch_size)

print(test_sum_num)
print(test_num_batches_per_epoch)
#exit()

#inputs = tf.placeholder(tf.float32, shape=[None, None, ctc._fea_dim], name="inputs1")
#labels = tf.sparse_placeholder(tf.int32, name="labels1")
#seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len1")

print("=============print placeholder name===============")

#print(inputs.name)
#print(seq_len.name)
#print(labels.name)

#exit()
tf.reset_default_graph()
#g = tf.get_default_graph()
sess = tf.Session()
#saver = tf.train.Saver()
#result = sess.run(y, feed_dict={inputs: data})
saver = tf.train.import_meta_graph("./thchs30/saved_model/saved_model_thchs30_2_4/model.ckpt.meta")
saver.restore(sess,"./thchs30/saved_model/saved_model_thchs30_2_4/model.ckpt")
#tf.reset_default_graph()
#graph = tf.get_default_graph()
inputs = tf.get_default_graph().get_tensor_by_name("inputs:0")
indices = tf.get_default_graph().get_tensor_by_name("labels/indices:0")
values = tf.get_default_graph().get_tensor_by_name("labels/values:0")
shape = tf.get_default_graph().get_tensor_by_name("labels/shape:0")
seq_len = tf.get_default_graph().get_tensor_by_name("seq_len:0")
cost = tf.get_default_graph().get_tensor_by_name("cost:0")
ler = tf.get_default_graph().get_tensor_by_name("ler:0")
test_cost = 0
#print('Successfully load the pre-trained model!')
#print(inputs.name)
#print(seq_len.name)
test_ler = 0
start = time.time()
print("begin debug...")
for test_batch in tqdm(range(test_num_batches_per_epoch)):
    test_start_idx = test_batch * ctc._batch_size
    test_end_idx = min(test_start_idx + ctc._batch_size, test_sum_num)
    #print(test_start_idx)
    #print(test_end_idx)
    #print("-"*20)
    test_inputs, test_labels, test_seq = ctc.next_batch(test_dic, test_start_idx, test_end_idx, is_training = "train")
    #print(test_labels)
    feed_test = {
            inputs: test_inputs,
            (indices,values,shape): (test_labels[0],test_labels[1],test_labels[2]),
            seq_len: test_seq
            }
    test_batch_cost, test_batch_ler = sess.run([cost,ler], feed_test)
    test_cost += test_batch_cost*ctc._batch_size
    test_ler += test_batch_ler*ctc._batch_size

test_cost = test_cost/test_sum_num
test_ler = test_ler/test_sum_num
#log = "Epoch {}/{}, dev_cost = {:.3f}, dev_ler = {:.3f}, time = {:.3f}"
delt_time = time.time() - start
print("run time:" + str(delt_time))
print("test_cost:" + str(test_cost))
print("test_ler:" + str(test_ler))
