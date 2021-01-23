import tensorflow as tf
import numpy as np
import time
import os
from tqdm import tqdm

import scipy.io.wavfile as wav
from python_speech_features import mfcc
from python_speech_features import logfbank
from python_speech_features import delta

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
#os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
#os.environ['CUDA_VISIBLE_DEVICES']="0,1"
#tf.device('/gpu:1')
class CTC:
	def __init__(self, num_epochs, num_units, num_layers, batch_size, learning_rate):
		self._num_epochs = num_epochs
		self._num_units = num_units
		self._num_layers = num_layers
		self._batch_size = batch_size
		self._learning_rate = learning_rate
		self._val_nums = 200
		
		self._vocab_char = {}
		self._vocab_phoneme = {}
		self._vocab_number_phone = {}
		self._char_num_class = 0
		self._phoneme_num_class = 0
				#exit()
		self._fea_dim = 13 # or 26
		self._temp_len = []
		self._temp_len_dev = []
		self._temp_len_test = []
	# Input:
	# mfcc: [frame, feature_dim] = [930, 13]
	def get_mfcc_fea(self,wave_file):
		fs, audio = wav.read(wave_file)
		mfcc_inputs = mfcc(audio, samplerate=fs)
		train_mfcc_inputs = np.asarray(mfcc_inputs[np.newaxis,:])
		#train_mfcc_inputs = np.asarray(mfcc_inputs)
		train_mfcc_inputs = (train_mfcc_inputs-np.mean(train_mfcc_inputs))/np.std(train_mfcc_inputs)
		train_seq_len = [train_mfcc_inputs.shape[1]]

		return train_mfcc_inputs, train_seq_len
  
    # fbank: [frame, feature_dim] = [930, 26]
	def get_fbank_fea(self, wave_file):
		fs,audio = wav.read(wave_file)
		fbank_inputs = logfbank(audio, samplerate=fs)
		train_fbank_inputs = np.asarray(fbank_inputs[np.newaxis,:])
		train_fbank_inputs = (train_fbank_inputs-np.mean(train_fbank_inputs))/np.std(train_fbank_inputs)
		train_seq_len = [train_fbank_inputs.shape[1]]
		
		return train_fbank_inputs, train_seq_len
	
	# Output:
    # review all phoneme file, deduplicate the same phoneme and map them to numbers(or tags) 
	def get_train_target_char(self, txt_file):
		#self.gen_vocab_char(scp_file)
		train_targets_char = []
		with open(txt_file, "r") as f:
			for i,j in enumerate(f.readlines()):
				if i==0:
					line = j.strip("\n ").split(" ")
					str_line = "".join(line)
					#print(str_line)
					for i in str_line:
						train_targets_char.append(self._vocab_char[i])
		#train_targets_char = np.array(train_targets_char)
		#print(type(train_targets_char))
		## such as: [array([])]
		#print(train_targets)
		#train_targets = sparse_tuple_from(train_targets)
		
		return train_targets_char

	# map word(include <BLANK>) to number
	def gen_vocab_char(self, scp_file):
		t = set()
		d = [(0,"<BLANK>")]
		dc = {}
		max_frame_char = 0
		with open(scp_file, "r") as f:
			for line in f.readlines():
				line = line.strip("\n ");
				list_s = []
				with open(line, "r") as lf:
					first = lf.readline()
					#second = lf.readline()
					#third = lf.readline()
					line = first.strip("\n ").split(" ")
					s = ""
					for l in line:
						s=s+l
					list_s = list(s)
				for item in list_s:
					t.add(item)
		# 以下可用zip函数替换，不必要写这么多
		for i,t in enumerate(t):
			d.append((i+1,t))
		for i in d:
			dc[i[1]] = i[0]
		self._vocab_char = dc
		self._char_num_class = len(dc)
	
	def get_train_target_phoneme(self, txt_file):
		#self.gen_vocab_phoneme(scp_file)
		train_targets_phoneme = []
		with open(txt_file, "r") as f:
			for i,j in enumerate(f.readlines()):
				if i==2:
					line = j.strip("\n ").split(" ")
					for i in line:
						train_targets_phoneme.append(self._vocab_phoneme[i])
		#train_targets_phoneme = np.array(train_targets_phoneme)	
		return train_targets_phoneme

	def gen_vocab_phoneme(self, scp_file):
		t = set()
		#d = [(0,"<BLANK>")]
		d = []
		dc_phone_number = {}
		dc_number_phone = {}
		with open(scp_file, "r") as f:
			for line in f.readlines():
				line = line.strip("\n ");
				with open(line, "r") as path_line:
					inner_line = "./thchs30/data_thchs30"+path_line.readline().strip("\n ")[2:]
				with open(inner_line, "r") as lf:
					first = lf.readline()
					second = lf.readline()
					third = lf.readline()
					line = third.strip("\n ").split(" ")
				for item in line:
					t.add(item)
		# 以下可用zip函数替换，不必要写这么多
		for i,t in enumerate(t):
			#d.append((i+1,t))
			d.append((i,t))
		for i in d:
			dc_phone_number[i[1]] = i[0]
			dc_number_phone[i[0]] = [i[1]]
		self._vocab_phoneme = dc_phone_number
		self._phoneme_num_class = len(dc_phone_number)+1
		self._vocab_number_phone = dc_number_phone
	
	def gen_vocab_phoneme_from_lexicon(self, lexicon_file):
		
		temp_list = []
		with open(lexicon_file, "r") as f:
			for line in f.readlines():
				line = line.strip("\n ").split(" ")
				if line[0] == "<eps>":
					continue
				temp_list.append(line[0])
		self._vocab_phoneme = dict(zip(temp_list, range(len(temp_list))))
		self._phoneme_num_class = len(self._vocab_phoneme)+1

	def data_handle(self, wav_scp_file, txt_scp_file, lexicon_file, is_training = "train"):
		# 生成phoneme dict phonme和number一一对应
		#self.gen_vocab_phoneme(txt_scp_file)
		self.gen_vocab_phoneme_from_lexicon(lexicon_file)
		# 针对each line 进行映射，放进label_list中
		inputs = []
		labels = []
		with open(txt_scp_file, "r") as label_files:
			for line in tqdm(label_files.readlines()):
				line = line.strip("\n")
				with open(line, "r") as path_line:
					inner_line = "./thchs30/data_thchs30"+path_line.readline().strip("\n ")[2:]
				#todo 调用get_train_target_phoneme获取每一个.trn文件中的标签即可
				labels.append(self.get_train_target_phoneme(inner_line))
		# 处理特征，并且进行paddding，便于input 对应tensorflow
		# mfcc特征:
		with open(wav_scp_file, "r") as wav_files:
			for line2 in tqdm(wav_files.readlines()):
				line2 = line2.strip("\n ")
				train_mfcc_inputs, train_seq_len = self.get_mfcc_fea(line2)
				inputs.append(train_mfcc_inputs)
		if is_training == "train":
			for i in range(len(inputs)):
				self._temp_len.append(inputs[i].shape[1])
			len_temp = np.array(self._temp_len)
		elif is_training == "dev":
			for i in range(len(inputs)):
				self._temp_len_dev.append(inputs[i].shape[1])
			len_temp = np.array(self._temp_len_dev)
		elif is_training == "test":
			for i in range(len(inputs)):
				self._temp_len_test.append(inputs[i].shape[1])
			len_temp = np.array(self._temp_len_test)
		for i,pad in enumerate(np.max(len_temp)-len_temp):
			inputs[i] = np.pad(inputs[i],((0,0),(0,pad),(0,0)),mode="constant", constant_values=0)
		dic_return = {
				"inputs":inputs,
				"labels":labels
				}
		return dic_return

	def next_batch(self, dic_return, start_idx, end_idx, is_training = "train"):
		seq_len = []
		inputs_x = dic_return["inputs"][start_idx:end_idx]
		inputs_y = dic_return["labels"][start_idx:end_idx]
		# sparse tuple:
		inputs_y = self.sparse_data(np.asarray(inputs_y))
		if is_training == "train":
			for i in range(start_idx, end_idx):
				seq_len.append(self._temp_len[i])
		elif is_training == "dev":
			for i in range(start_idx, end_idx):
				seq_len.append(self._temp_len_dev[i])
		elif is_training == "test":
			for i in range(start_idx, end_idx):
				seq_len.append(self._temp_len_test[i])
		'''
		seq_len_np = np.array(seq_len)
		for i,pad in enumerate(np.max(seq_len_np)-seq_len_np):
			inputs_x[i] = np.pad(inputs_x[i],((0,0),(0,pad),(0,0)),mode="constant", constant_values=0)
		inputs_x = np.concatenate(inputs_x, axis=0)
		#todo: how to padding -> done.
		'''
		inputs_x = np.concatenate(inputs_x, axis=0)
		return inputs_x, inputs_y, seq_len
	
	def sparse_data(self, seq, dtype=np.int32):
		"""
		为了适应ctc_loss中的sparseTensor
		
		input: np.array

		return: tuple(indices,values,shape)
		
		"""
		indices = []
		values = []
		for n, s in enumerate(seq):
			indices.extend(zip([n] * len(s), range(len(s))))
			values.extend(s)

		indices = np.asarray(indices, dtype=np.int64)
		values = np.asarray(values, dtype=dtype)
		shape = np.asarray([len(seq), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
		
		return indices, values, shape

if __name__ == "__main__":
	print("--"*10 + "programming begin" + "--"*10 )
	print("loading data...")
	wav_scp_file = "./scp/inputs.scp"
	txt_scp_file = "./scp/labels.scp"
	dev_wav_scp_file = "./scp/dev_inputs.scp"
	dev_txt_scp_file = ".scp/dev_lables.scp"
	lexicon_file = "./thchs30/data_thchs30/lm_phone/lexicon.txt"
	print("loading done.")
	# num_epochs, num_units, num_layers, batch_size, learning_rate
	print("initializing...")
	# 34 epoch 512 2 8 0.0001: train_cost 64.074 train_ler = 0.172
	ctc = CTC(800,128,2,16,0.0001) # initial __init__ para
	dic_return = ctc.data_handle(wav_scp_file, txt_scp_file, lexicon_file)
	dev_dic = ctc.data_handle(dev_wav_scp_file, dev_txt_scp_file, lexicon_file, is_training = "dev")
	'''
	with open("./vocab.txt","a") as vocab_file:
		for i in ctc._vocab_phoneme.items():
			vocab_file.write(i[0]+":"+str(i[1])+"\n")
	'''
	inputs_re = dic_return["inputs"]
	inputs_dev = dev_dic["inputs"]
	#print(np.shape(inputs_re))
	#labels_re = dic_return["labels"]
	train_sum_num = len(inputs_re)
	dev_sum_num = len(inputs_dev)
	num_batches_per_epoch = int(train_sum_num/ctc._batch_size)
	dev_num_batches_per_epoch = int(dev_sum_num/ctc._batch_size)
	# 计算图
	print("loading graph...")
	g = tf.Graph()
	with g.as_default():
		# todo input: for ctc loss parameters done.
		inputs = tf.placeholder(tf.float32, shape=[None, None, ctc._fea_dim], name="inputs")
		# some questions?done.: [batch_size] each dim represents frame_nums
		labels = tf.sparse_placeholder(tf.int32,name="labels")
		seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len")
		# sparse for ctc_loss
		cells = []
		for i in range(ctc._num_layers):
			cell = tf.contrib.rnn.LSTMCell(num_units=ctc._num_units, use_peepholes=False, state_is_tuple=True)
			cells.append(cell)
		# time_major False: (batch, time step, input); True: (time step, batch, input)
		# outputs: [batch_size, max_time, cell.output_size] cell.output_size = BasicLSTMCel中的num_units
		mcell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
		outputs, states = tf.nn.dynamic_rnn(cell=mcell, inputs=inputs, sequence_length=seq_len, dtype=tf.float32, time_major=False)
		# 将outputs的三维转化为二维数据记性权重W和b的相乘（不然乘不了）
		outputs = tf.reshape(outputs, [-1,ctc._num_units], name="outputs")
		# W & b:
		W = tf.Variable(tf.truncated_normal(shape=[ctc._num_units,ctc._phoneme_num_class], mean=0, stddev=0.1),name="W")
		b = tf.Variable(tf.constant(0., shape=[ctc._phoneme_num_class]),name="b")

		lgts = tf.matmul(outputs, W) + b
		#-----------------------#: some error accur: because the last batch size < ctc._batch_size
		batch_size_temp = tf.shape(inputs)[0]
		lgts = tf.reshape(lgts, [batch_size_temp, -1, ctc._phoneme_num_class], name="lgts")
		#lgts = tf.reshape(lgts, [ctc._batch_size, -1, ctc._phoneme_num_class])
		# 在此未使用softmax,而是直接将lgts作为ctc loss的输入 changed.
		#lgts = tf.nn.softmax(lgts)
		# ctc loss
		########################### todo:lable = SparseTensor######################
		loss = tf.nn.ctc_loss(labels=labels, inputs=lgts, sequence_length=seq_len, time_major=False)
		cost = tf.reduce_mean(loss,name="cost")
		trainer = tf.train.AdamOptimizer(learning_rate=ctc._learning_rate).minimize(cost)
		# according to original paper's para
		#trainer = tf.train.MomentumOptimizer(learning_rate=ctc._learning_rate, momentum=0.9).minimize(cost)
		# greedy search or bear search: 对过完rnn的output token进行decoder
		lgts_reverse = tf.transpose(lgts,(1,0,2))
		#decoded, log_prob = tf.nn.ctc_beam_search_decoder()
		decoded, log_prob = tf.nn.ctc_greedy_decoder(lgts_reverse, sequence_length=seq_len, merge_repeated=True)
		# according to paper: label error rate
		ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels),name="ler")
		#dense_decoded = tf.sparse_to_dense(decoded[0])
		#modify : add max_to_keep=1
		saver = tf.train.Saver(max_to_keep=1)
	# feed data
	print("begin trainning...")
	config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
	#config.gpu_options.per_process_gpu_memory_fraction = 0.9 # maximun alloc gpu50% of MEM
	#config.gpu_options.allow_growth = True #allocate dynamically
	with tf.Session(graph=g, config=config) as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in tqdm(range(ctc._num_epochs)):
			train_cost = 0
			train_ler = 0
			dev_cost = 0
			dev_ler = 0
			start = time.time()
			for dev_batch in tqdm(range(dev_num_batches_per_epoch)):
				dev_start_idx = dev_batch * ctc._batch_size
				dev_end_idx = min(dev_start_idx + ctc._batch_size, dev_sum_num)
				dev_inputs, dev_labels, dev_seq = ctc.next_batch(dev_dic, dev_start_idx, dev_end_idx, is_training = "dev")
				feed_dev = {
						inputs: dev_inputs,
						labels: dev_labels,
						seq_len: dev_seq
						}
				dev_batch_cost, dev_batch_ler = sess.run([cost,ler], feed_dev)
				dev_cost += dev_batch_cost*ctc._batch_size
				dev_ler += dev_batch_ler*ctc._batch_size
			for batch in tqdm(range(num_batches_per_epoch)):
				start_idx = batch * ctc._batch_size
				end_idx = min(start_idx + ctc._batch_size,train_sum_num)
				inputs_x, inputs_y, seq_len_batch = ctc.next_batch(dic_return, start_idx, end_idx)
				#aha = np.shape(inputs_x)
				#print(aha[0])
				feed = {
						inputs: inputs_x,
						labels: inputs_y,
						seq_len: seq_len_batch
						}
				#print("feed done.")
				batch_cost, _, batch_ler = sess.run([cost, trainer, ler], feed)
				train_cost += batch_cost*ctc._batch_size
				train_ler += batch_ler*ctc._batch_size
				# todo: dev for every epoch: feed diff data in
			dev_cost = dev_cost/dev_sum_num
			dev_ler = dev_ler/dev_sum_num
			# SparseTensor2Dense:
			# def tf.sparse_tensor_to_dense(sp_input,default_value=0,validate_indices=True,name=None):
			# shuffle data:利用索引对行的全排列进行shuffle
			# 注意在shuffle之前要转化为ndarray，在转化之前需要保证是pad过的，不然会报错
			shuffle_indices = np.random.permutation(train_sum_num)
			dic_return["inputs"] = np.array(dic_return["inputs"])[shuffle_indices]
			dic_return["labels"] = np.array(dic_return["labels"])[shuffle_indices]	
			train_cost = train_cost/train_sum_num
			train_ler = train_ler/train_sum_num
			log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, dev_cost = {:.3f}, dev_ler = {:.3f}, time = {:.3f}"
			# todo : log file
			print(log.format(epoch+1, ctc._num_epochs, train_cost, train_ler, dev_cost, dev_ler, time.time()-start))
			with open("./ctc3.log", "a") as f:
				f.write(log.format(epoch+1, ctc._num_epochs,train_cost, train_ler, dev_cost, dev_ler, time.time()-start)+ "\n")
		save_path = saver.save(sess, './thchs30/saved_model/saved_model_thchs30_2_4/model.ckpt')
		print('model saved.')
