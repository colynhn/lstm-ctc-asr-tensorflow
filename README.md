An baseline implementation of end2end asr using tensorflow in THCHS30 datasets 

## Requirements ##

* Tensorflow 1.12
* numpy 1.19
* tqdm 4.47
* scipy 1.5
* python_speech_features 0.6 

## Download thchs30 dataset ##
you can
```
bash run.sh
```
or 

visiting http://www.openslr.org/resources/18
## Process data ##

generate train/dev/test scp file

```
python gen_trn_scp.py [thchs30 train wav dir] [wav.scp file] [trn.scp file]
python gen_trn_scp.py [thchs30 dev wav dir] [wav.scp file] [trn.scp file]
python gen_trn_scp.py [thchs30 test wav dir] [wav.scp file] [trn.scp file]
```

## Train & Test & Recognize ##

before train/test/recognize you'd better to verify some file paths in ctc_train.py 
such as the saved path of model you have trained and so on when you can not run the program.  

Train

```
python ctc_train.py
```

Test

```
python ctc_test.py
```

Recognize

```
python recognize.py
```

lacking of phonme vocab,the recognize.py can only generate phoneme sequence,you can have a further try.

## Reference ##

https://github.com/igormq/ctc_tensorflow_example




```
peace & love
```
