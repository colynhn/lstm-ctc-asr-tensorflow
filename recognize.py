import tensorflow as tf
from tqdm import tqdm
import numpy as np
import time
import scipy.io.wavfile as wav
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

from ctc_train3 import CTC
ctc = CTC(800,128,2,16,0.0001)


test_wav_scp_file = "./scp/test_wav_scp_file1"
lexicon_file = "./thchs30/data_thchs30/lm_phone/lexicon.txt"
print("loading source data done.")

train_inputs =  []
train_temp_len = []

with open(test_wav_scp_file, "r") as wav_files:
    for line2 in tqdm(wav_files.readlines()):
        line2 = line2.strip("\n ")
        train_mfcc_inputs, train_seq_len = ctc.get_mfcc_fea(line2)
        train_inputs.append(train_mfcc_inputs)
        train_temp_len.append(train_seq_len)

ctc.gen_vocab_phoneme_from_lexicon(lexicon_file)

tf.reset_default_graph()
sess = tf.Session()
saver = tf.train.import_meta_graph("./thchs30/saved_model/saved_model_thchs30_2_4/model.ckpt.meta")
saver.restore(sess,"./thchs30/saved_model/saved_model_thchs30_2_4/model.ckpt")

inputs = tf.get_default_graph().get_tensor_by_name("inputs:0")
seq_len = tf.get_default_graph().get_tensor_by_name("seq_len:0")
lgts = tf.get_default_graph().get_tensor_by_name("lgts:0")
lgts = tf.reshape(lgts, [1, -1, ctc._phoneme_num_class])
lgts_reverse = tf.transpose(lgts,(1,0,2))
decoded, log_prob = tf.nn.ctc_greedy_decoder(lgts_reverse, sequence_length=seq_len, merge_repeated=True)
result = tf.sparse_tensor_to_dense(decoded[0])

num_to_phone = dict()

for key,value in ctc._vocab_phoneme.items():
    num_to_phone[value] = key


start = time.time()

for i in range(len(train_inputs)):

    feed_test = {
            inputs: train_inputs[i],
            seq_len:train_temp_len[i]
            }
    result = sess.run(result, feed_test)
    delt_time = time.time() - start
    print("run time:" + str(delt_time))
    ret = []
    for i in result:
        for j in i:
            ret.append(num_to_phone[j])
    print(ret)
