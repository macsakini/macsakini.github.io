---
classes: wide
---

``` python
!pip install -q tf-models-official
!pip install -q tensorflow-text
!pip install -q tf-models-official==2.3.0Sa
```

    [K     |████████████████████████████████| 1.1MB 6.9MB/s 
    [K     |████████████████████████████████| 51kB 6.8MB/s 
    [K     |████████████████████████████████| 37.6MB 124kB/s 
    [K     |████████████████████████████████| 174kB 50.3MB/s 
    [K     |████████████████████████████████| 1.2MB 46.8MB/s 
    [K     |████████████████████████████████| 276kB 41.9MB/s 
    [K     |████████████████████████████████| 102kB 12.0MB/s 
    [K     |████████████████████████████████| 358kB 44.5MB/s 
    [?25h  Building wheel for seqeval (setup.py) ... [?25l[?25hdone
      Building wheel for pyyaml (setup.py) ... [?25l[?25hdone
      Building wheel for py-cpuinfo (setup.py) ... [?25l[?25hdone
    [K     |████████████████████████████████| 3.4MB 5.0MB/s 
    [K     |████████████████████████████████| 849kB 5.4MB/s 
    [?25h


``` python
import requests
r = requests.post("http://3386c69248d9.ngrok.io/", data={'foo': 'The Role of Saying No is sometimes seen as a luxury that only those in power can afford . But saying no is not merely a privilege reserved for the successful among us . It is also a strategy that can help you become successful . Steve Jobs famously said, “People think focus means saying yes to the thing you’ve got to focus on. But that’s not what it means at all. It means saying no to the hundred other good ideas that there are. You have to pick carefully’re not always saying yes,” Steve Jobs said. “If you don’t guard your time, people will steal it from you," says Pedro Sorrentino. ‘If you are not guarding your time.’ says Sorrentinos.‘ If you want to say no to distractions, it means you need to say yes, it is the only productivity hack,’ he says. You may have to try many things to discover what works'})
# And done.
print(r.text) # displays the result body.
```

     The Role of Saying No is sometimes seen as a luxury that only those in power can afford . But saying no is not merely a privilege reserved for the successful among us . It is also a strategy that can help you become successful



```python
#Import dependencies
#Import necessary dependancies
import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks

from google.colab import files
from google.colab import drive
import pandas as pd
import io
import numpy

```


```python
gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"
tf.io.gfile.listdir(gs_folder_bert)
```




    ['bert_config.json',
     'bert_model.ckpt.data-00000-of-00001',
     'bert_model.ckpt.index',
     'vocab.txt']




```
hub_url_bert = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
```


```
uploaded = files.upload()

dataset = pd.read_csv(io.BytesIO(uploaded['profitaa.csv']))

df2 = dataset.sample(frac=0.8, random_state=0)
df2_test = dataset.drop(df2.index)

```



<input type="file" id="files-020ab0fe-2813-4061-becf-21020407b709" name="files[]" multiple disabled
   style="border:none" />
<output id="result-020ab0fe-2813-4061-becf-21020407b709">
 Upload widget is only available when the cell has been executed in the
 current browser session. Please rerun this cell to enable.
 </output>
 <script src="/nbextensions/google.colab/files.js"></script> 


    Saving summary.csv to summary.csv



```
df2.shape

df2["Relevancy_Score"].isnull().values.any()

df3 = df2.dropna()

df3.describe().transpose()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>headline</th>
      <td>1487</td>
      <td>1487</td>
      <td>\nCreate a comfortable home for both hamsters....</td>
      <td>1</td>
    </tr>
    <tr>
      <th>title</th>
      <td>1487</td>
      <td>1487</td>
      <td>How to Tell a Middle School Boy You Like Him</td>
      <td>1</td>
    </tr>
    <tr>
      <th>text</th>
      <td>1487</td>
      <td>1486</td>
      <td>,,</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3["Relevancy_Score"].isnull().values.any()

df3_test = df2_test.dropna()
```


```

```


```python
# Set up tokenizer to generate Tensorflow dataset
tokenizer = bert.tokenization.FullTokenizer(
    vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
     do_lower_case=True)s

print("Vocab size:", len(tokenizer.vocab))
```

    Vocab size: 30522



```python
tokens = tokenizer.tokenize("Hello TensorFlow!")
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
```

    ['hello', 'tensor', '##flow', '!']
    [7592, 23435, 12314, 999]



```python
tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
```




    [101, 102]




```python
def encode_sentencer(s):
   tokens = list(tokenizer.tokenize(s))
   tokens.append('[SEP]')
   return tokenizer.convert_tokens_to_ids(tokens)

sentence1 = tf.ragged.constant([
    encode_sentencer(s) for s in df3["Sentence_1"]])
sentence2 = tf.ragged.constant([
    encode_sentencer(s) for s in df3["Sentence_2"]])
```


```python
print("Sentence1 shape:", sentence1.shape.as_list())
print("Sentence2 shape:", sentence2.shape.as_list())
print(sentence1[0])
```

    Sentence1 shape: [1487, None]
    Sentence2 shape: [1487, None]
    tf.Tensor(
    [ 4638  2065  1996 10654  6238  1005  1055  4373  2203  2003  4954  1012
      1010  3198  1996  9004  4497  1013  8843  2121  2065  2017  2064  5047
      1996 10654 15608  1012  1010  2298  2005 10654 15608  2008  2031 12538
     15695  1010  4408  2159  1010  1998  2024  3227  5379  1010  2065  1996
     10654  6238  2003  1037  2978  5376  2100  2012  2034  2043  2017  5047
      2032  1013  2014  1010  2123  1005  1056  4737  2008  2003  2025  1037
      2919  2518  1012  1010  5454  1037 10654  6238  2008  2017  2066  1012
       102], shape=(85,), dtype=int32)



```python
cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]
input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)
_ = plt.pcolormesh(input_word_ids.to_tensor())
```


<img src="/assets/images/2022-05-03-BERT-Analysis_files/2022-05-03-BERT-Analysis_14_0.png">



```python
input_mask = tf.ones_like(input_word_ids).to_tensor()

plt.pcolormesh(input_mask)
```




    <matplotlib.collections.QuadMesh at 0x7faa562b40f0>




<img src="/assets/images/2022-05-03-BERT-Analysis_files/2022-05-03-BERT-Analysis_15_1.png">



```python
type_cls = tf.zeros_like(cls)
type_s1 = tf.zeros_like(sentence1)
type_s2 = tf.ones_like(sentence2)
input_type_ids = tf.concat([type_cls, type_s1, type_s2], axis=-1).to_tensor()

plt.pcolormesh(input_type_ids)
```




    <matplotlib.collections.QuadMesh at 0x7faa56299320>




<img src="/assets/images/2022-05-03-BERT-Analysis_files/2022-05-03-BERT-Analysis_16_1.png">



```python
def encode_sentence(s, tokenizer):
   tokens = list(tokenizer.tokenize(s))
   tokens.append('[SEP]')
   return tokenizer.convert_tokens_to_ids(tokens)


def bert_encode(glue_dict, tokenizer):
  num_examples = len(glue_dict["Sentence_1"])
  
  sentence1 = tf.ragged.constant([
      encode_sentence(s, tokenizer)
      for s in np.array(glue_dict["Sentence_1"])])
  sentence2 = tf.ragged.constant([
      encode_sentence(s, tokenizer)
       for s in np.array(glue_dict["Sentence_2"])])

  cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]
  input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

  input_mask = tf.ones_like(input_word_ids).to_tensor()

  type_cls = tf.zeros_like(cls)
  type_s1 = tf.zeros_like(sentence1)
  type_s2 = tf.ones_like(sentence2)
  input_type_ids = tf.concat(
      [type_cls, type_s1, type_s2], axis=-1).to_tensor()

  inputs = {
      'input_word_ids': input_word_ids.to_tensor(),
      'input_mask': input_mask,
      'input_type_ids': input_type_ids}

  return inputs
```


```python
df_train = bert_encode(df3, tokenizer)
#df_labels = df3['Relevancy_Score'].div(5)

df_test = bert_encode(df3_test, tokenizer)
df_test_labels = df3_test['Relevancy_Score'].div(5)

print(df_train)
print(df_test)
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    /usr/local/lib/python3.6/dist-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2897             try:
    -> 2898                 return self._engine.get_loc(casted_key)
       2899             except KeyError as err:


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 'Sentence_1'

    
    The above exception was the direct cause of the following exception:


    KeyError                                  Traceback (most recent call last)

    <ipython-input-29-1e8639691852> in <module>()
    ----> 1 df_train = bert_encode(df3, tokenizer)
          2 #df_labels = df3['Relevancy_Score'].div(5)
          3 
          4 df_test = bert_encode(df3_test, tokenizer)
          5 df_test_labels = df3_test['Relevancy_Score'].div(5)


    <ipython-input-27-02cbe3d43c41> in bert_encode(glue_dict, tokenizer)
          6 
          7 def bert_encode(glue_dict, tokenizer):
    ----> 8   num_examples = len(glue_dict["Sentence_1"])
          9 
         10   sentence1 = tf.ragged.constant([


    /usr/local/lib/python3.6/dist-packages/pandas/core/frame.py in __getitem__(self, key)
       2904             if self.columns.nlevels > 1:
       2905                 return self._getitem_multilevel(key)
    -> 2906             indexer = self.columns.get_loc(key)
       2907             if is_integer(indexer):
       2908                 indexer = [indexer]


    /usr/local/lib/python3.6/dist-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2898                 return self._engine.get_loc(casted_key)
       2899             except KeyError as err:
    -> 2900                 raise KeyError(key) from err
       2901 
       2902         if tolerance is not None:


    KeyError: 'Sentence_1'



```python
for key, value in df_train.items():
  print(f'{key:15s} shape: {value.shape}')

print(f'df_labels shape: {df_labels.shape}')
```

    input_word_ids  shape: (2396, 125)
    input_mask      shape: (2396, 125)
    input_type_ids  shape: (2396, 125)
    df_labels shape: (2396,)



```python
import json

bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())

bert_config = bert.configs.BertConfig.from_dict(config_dict)

config_dict
```




    {'attention_probs_dropout_prob': 0.1,
     'hidden_act': 'gelu',
     'hidden_dropout_prob': 0.1,
     'hidden_size': 768,
     'initializer_range': 0.02,
     'intermediate_size': 3072,
     'max_position_embeddings': 512,
     'num_attention_heads': 12,
     'num_hidden_layers': 12,
     'type_vocab_size': 2,
     'vocab_size': 30522}




```
print(bert.bert_models)
bert_classifier, bert_encoder = bert.bert_models.classifier_model(
   bert_config, num_labels=1)
```

    <module 'official.nlp.bert.bert_models' from '/usr/local/lib/python3.6/dist-packages/official/nlp/bert/bert_models.py'>



```
tf.keras.utils.plot_model(bert_classifier, show_shapes=True, dpi=48)
```


```
glue_batch = {key: val[:10] for key, val in df_train.items()}

bert_classifier(
    glue_batch, training=True
).numpy()
```




    array([[-0.17102675],
           [ 0.07586445],
           [ 0.01797828],
           [-0.19046766],
           [-0.06210539],
           [-0.09033417],
           [ 0.01831295],
           [-0.08006046],
           [-0.22937882],
           [-0.03147416]], dtype=float32)




```
tf.keras.utils.plot_model(bert_encoder, show_shapes=True, dpi=48)
```


```
[print(i.shape, i.dtype) for i in bert_classifier.inputs]
[print(o.shape, o.dtype) for o in bert_classifier.outputs]
[print(l.name, l.input_shape, l.dtype) for l in bert_classifier.layers]
bert_classifier.summary()
```

    (None, None) <dtype: 'int32'>
    (None, None) <dtype: 'int32'>
    (None, None) <dtype: 'int32'>
    (None, 1) <dtype: 'float32'>
    input_word_ids [(None, None)] int32
    input_mask [(None, None)] int32
    input_type_ids [(None, None)] int32
    transformer_encoder [(None, None), (None, None), (None, None)] float32
    dropout_1 (None, 768) float32
    classification (None, 768) float32
    Model: "bert_classifier"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_word_ids (InputLayer)     [(None, None)]       0                                            
    __________________________________________________________________________________________________
    input_mask (InputLayer)         [(None, None)]       0                                            
    __________________________________________________________________________________________________
    input_type_ids (InputLayer)     [(None, None)]       0                                            
    __________________________________________________________________________________________________
    transformer_encoder (Transforme [(None, None, 768),  109482240   input_word_ids[0][0]             
                                                                     input_mask[0][0]                 
                                                                     input_type_ids[0][0]             
    __________________________________________________________________________________________________
    dropout_1 (Dropout)             (None, 768)          0           transformer_encoder[0][1]        
    __________________________________________________________________________________________________
    classification (Classification) (None, 1)            769         dropout_1[0][0]                  
    ==================================================================================================
    Total params: 109,483,009
    Trainable params: 109,483,009
    Non-trainable params: 0
    __________________________________________________________________________________________________



```
checkpoint = tf.train.Checkpoint(model=bert_encoder)
checkpoint.restore(
    os.path.join(gs_folder_bert, 'bert_model.ckpt')).assert_consumed()
```




    <tensorflow.python.training.tracking.util.CheckpointLoadStatus at 0x7fe309295a20>




```
# Set up epochs and steps
epochs = 2
batch_size = 32
eval_batch_size = 32

train_data_size = len(df_labels)
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

# creates an optimizer with learning rate schedule
optimizer = nlp.optimization.create_optimizer(
    2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)
```


```
type(optimizer)
```




    official.nlp.optimization.AdamWeightDecay




```
metric = [tf.keras.metrics.Accuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.MeanAbsoluteError()

bert_classifier.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['mse', 'mae', 'mape', 'cosine_similarity','accuracy'])

bert_classifier.fit(
      df_train, df_labels,
      batch_size=32,
      epochs=epochs)
```

    Epoch 1/2
    75/75 [==============================] - 3287s 44s/step - loss: 0.3301 - mse: 0.1647 - mae: 0.3301 - mape: 26140062.6908 - cosine_similarity: 0.8531 - accuracy: 0.0810
    Epoch 2/2
    75/75 [==============================] - 3245s 43s/step - loss: 0.1562 - mse: 0.0406 - mae: 0.1562 - mape: 11219197.4079 - cosine_similarity: 0.8746 - accuracy: 0.1360





    <tensorflow.python.keras.callbacks.History at 0x7fe3034ba908>




```
export_dir='./saved_model2'
tf.saved_model.save(bert_classifier, export_dir=export_dir)

```

    WARNING:absl:Found untraced functions such as self_attention_layer_call_fn, self_attention_layer_call_and_return_conditional_losses, attention_output_layer_call_fn, attention_output_layer_call_and_return_conditional_losses, dropout_1_layer_call_fn while saving (showing 5 of 840). These functions will not be directly callable after loading.
    WARNING:absl:Found untraced functions such as self_attention_layer_call_fn, self_attention_layer_call_and_return_conditional_losses, attention_output_layer_call_fn, attention_output_layer_call_and_return_conditional_losses, dropout_1_layer_call_fn while saving (showing 5 of 840). These functions will not be directly callable after loading.


    INFO:tensorflow:Assets written to: ./saved_model2/assets


    INFO:tensorflow:Assets written to: ./saved_model2/assets



```

```


```
bert_classifier.evaluate(df_test,df_test_labels)
```

    19/19 [==============================] - 172s 9s/step - loss: 0.1405 - mse: 0.0355 - mae: 0.1405 - mape: 12436080.0000 - cosine_similarity: 0.8980 - accuracy: 0.1405





    [0.140456423163414,
     0.03549607843160629,
     0.140456423163414,
     12436080.0,
     0.8979933261871338,
     0.14046822488307953]




```
my_examples = bert_encode(
    glue_dict = {
        'Sentence_1':[
            'The rain in Spain falls mainly on the plain.',
            'Look I fine tuned BERT.',
            'I am alive.'],
        'Sentence_2':[
            'It mostly rains on the flat lands of Spain.',
            'Is it working? This does not match.',
            "I am alive."]
    },
    tokenizer=tokenizer)
result = bert_classifier(my_examples, training=False)

print(result)

#result = tf.argmax(result).numpy()

array = result.numpy()

def normalize(value):
	normalized = (value + 1) / (2);
	return normalized;


for i in array:
  x = normalize(i)
  print()
```

    tf.Tensor(
    [[0.6340052]
     [0.1166743]
     [0.7393124]], shape=(3, 1), dtype=float32)
    
    
    



```
!pip install h5py
```

    Requirement already satisfied: h5py in /usr/local/lib/python3.6/dist-packages (2.10.0)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from h5py) (1.15.0)
    Requirement already satisfied: numpy>=1.7 in /usr/local/lib/python3.6/dist-packages (from h5py) (1.19.5)



```
bert_classifier.save("./model.h5")
print("Saved model to disk")
```

    Saved model to disk



```
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```
reloaded = tf.saved_model.load(export_dir)
reloaded_result = reloaded([my_examples['input_word_ids'],
                            my_examples['input_mask'],
                            my_examples['input_type_ids']], training=False)

original_result = bert_classifier(my_examples, training=False)

# The results are (nearly) identical:
print(original_result.numpy())
print()
print(reloaded_result.numpy())
```

    [[0.6340052]
     [0.1166743]
     [0.7393124]]
    
    [[0.63400537]
     [0.11667421]
     [0.7393121 ]]



```

```


```
uploaded2 = files.upload()


predict = pd.read_csv(io.BytesIO(uploaded2['valtest.csv']))

predicter = bert_encode(glue_dict={"Sentence_1":predict['Sentence_1'],'Sentence_2': predict["Sentence_2"]}, tokenizer = tokenizer)

solutions = bert_classifier.predict(predicter)
```



<input type="file" id="files-978f0911-aea2-45bc-b283-f1a2353214f3" name="files[]" multiple disabled
   style="border:none" />
<output id="result-978f0911-aea2-45bc-b283-f1a2353214f3">
 Upload widget is only available when the cell has been executed in the
 current browser session. Please rerun this cell to enable.
 </output>
 <script src="/nbextensions/google.colab/files.js"></script> 


    Saving valtest.csv to valtest.csv



```
#a = (df.abs())

#b = normalized_predict

diff_pred = (solutions.tolist())

#print(diff_pred)

new_solutions = []
for i in diff_pred:
  for x in i:
    new_solutions.append(x)

ans = pd.Series(new_solutions)-predict['Relevancy_Score'].div(5)

print(ans.abs().describe().transpose())


```

    count    750.000000
    mean       0.126923
    std        0.096725
    min        0.000204
    25%        0.052139
    50%        0.108334
    75%        0.178874
    max        0.557846
    dtype: float64



```

bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8' 

map_name_to_handle = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_base/2',
    'electra_small':
        'https://tfhub.dev/google/electra_small/2',
    'electra_base':
        'https://tfhub.dev/google/electra_base/2',
    'experts_pubmed':
        'https://tfhub.dev/google/experts/bert/pubmed/2',
    'experts_wiki_books':
        'https://tfhub.dev/google/experts/bert/wiki_books/2',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
}

map_model_to_preprocess = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/2',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/2',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_preprocess/2',
    'electra_small':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'electra_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'experts_pubmed':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'experts_wiki_books':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
}

tfhub_handle_encoder = map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')


```

    BERT model selected           : https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1
    Preprocess model auto-selected: https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2



```
#Using the PREPROCESSING Model
bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
text_test = ['this is such an amazing movie!', 'I am Alive. I am Indebted','Praise to God almighty']
text_preprocessed = bert_preprocess_model(df2.Sentence_1)

print(text_preprocessed)

print(f'Keys       : {list(text_preprocessed.keys())}')
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

#Using the BERT Model
bert_model = hub.KerasLayer(tfhub_handle_encoder)
bert_results = bert_model(text_preprocessed)

print(bert_results)
print(f'Loaded BERT: {tfhub_handle_encoder}')
print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-6-d818c6d3088a> in <module>()
          2 bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
          3 text_test = ['this is such an amazing movie!', 'I am Alive. I am Indebted','Praise to God almighty']
    ----> 4 text_preprocessed = bert_preprocess_model(df2.Sentence_1)
          5 
          6 print(text_preprocessed)


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/engine/base_layer.py in __call__(self, *args, **kwargs)
       1010         with autocast_variable.enable_auto_cast_variables(
       1011             self._compute_dtype_object):
    -> 1012           outputs = call_fn(inputs, *args, **kwargs)
       1013 
       1014         if self._activity_regularizer:


    /usr/local/lib/python3.6/dist-packages/tensorflow_hub/keras_layer.py in call(self, inputs, training)
        235       result = smart_cond.smart_cond(training,
        236                                      lambda: f(training=True),
    --> 237                                      lambda: f(training=False))
        238 
        239     # Unwrap dicts returned by signatures.


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/smart_cond.py in smart_cond(pred, true_fn, false_fn, name)
         54       return true_fn()
         55     else:
    ---> 56       return false_fn()
         57   else:
         58     return control_flow_ops.cond(pred, true_fn=true_fn, false_fn=false_fn,


    /usr/local/lib/python3.6/dist-packages/tensorflow_hub/keras_layer.py in <lambda>()
        235       result = smart_cond.smart_cond(training,
        236                                      lambda: f(training=True),
    --> 237                                      lambda: f(training=False))
        238 
        239     # Unwrap dicts returned by signatures.


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/saved_model/load.py in _call_attribute(instance, *args, **kwargs)
        666 
        667 def _call_attribute(instance, *args, **kwargs):
    --> 668   return instance.__call__(*args, **kwargs)
        669 
        670 


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/def_function.py in __call__(self, *args, **kwds)
        826     tracing_count = self.experimental_get_tracing_count()
        827     with trace.Trace(self._name) as tm:
    --> 828       result = self._call(*args, **kwds)
        829       compiler = "xla" if self._experimental_compile else "nonXla"
        830       new_tracing_count = self.experimental_get_tracing_count()


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/def_function.py in _call(self, *args, **kwds)
        869       # This is the first call of __call__, so we have to initialize.
        870       initializers = []
    --> 871       self._initialize(args, kwds, add_initializers_to=initializers)
        872     finally:
        873       # At this point we know that the initialization is complete (or less


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/def_function.py in _initialize(self, args, kwds, add_initializers_to)
        724     self._concrete_stateful_fn = (
        725         self._stateful_fn._get_concrete_function_internal_garbage_collected(  # pylint: disable=protected-access
    --> 726             *args, **kwds))
        727 
        728     def invalid_creator_scope(*unused_args, **unused_kwds):


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/function.py in _get_concrete_function_internal_garbage_collected(self, *args, **kwargs)
       2967       args, kwargs = None, None
       2968     with self._lock:
    -> 2969       graph_function, _ = self._maybe_define_function(args, kwargs)
       2970     return graph_function
       2971 


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/function.py in _maybe_define_function(self, args, kwargs)
       3312     if self.input_signature is None or args is not None or kwargs is not None:
       3313       args, kwargs, flat_args, filtered_flat_args = \
    -> 3314           self._function_spec.canonicalize_function_inputs(*args, **kwargs)
       3315     else:
       3316       flat_args, filtered_flat_args = [None], []


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/function.py in canonicalize_function_inputs(self, *args, **kwargs)
       2695 
       2696     if self._input_signature is None:
    -> 2697       inputs, flat_inputs, filtered_flat_inputs = _convert_numpy_inputs(inputs)
       2698       kwargs, flat_kwargs, filtered_flat_kwargs = _convert_numpy_inputs(kwargs)
       2699       return (inputs, kwargs, flat_inputs + flat_kwargs,


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/function.py in _convert_numpy_inputs(inputs)
       2755         raise TypeError("The output of __array__ must be an np.ndarray "
       2756                         "(got {} from {}).".format(type(a), type(value)))
    -> 2757       flat_inputs[index] = constant_op.constant(a)
       2758       filtered_flat_inputs.append(flat_inputs[index])
       2759       need_packing = True


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/constant_op.py in constant(value, dtype, shape, name)
        263   """
        264   return _constant_impl(value, dtype, shape, name, verify_shape=False,
    --> 265                         allow_broadcast=True)
        266 
        267 


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/constant_op.py in _constant_impl(value, dtype, shape, name, verify_shape, allow_broadcast)
        274       with trace.Trace("tf.constant"):
        275         return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    --> 276     return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
        277 
        278   g = ops.get_default_graph()


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/constant_op.py in _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
        299 def _constant_eager_impl(ctx, value, dtype, shape, verify_shape):
        300   """Implementation of eager constant."""
    --> 301   t = convert_to_eager_tensor(value, ctx, dtype)
        302   if shape is None:
        303     return t


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/constant_op.py in convert_to_eager_tensor(value, ctx, dtype)
         96       dtype = dtypes.as_dtype(dtype).as_datatype_enum
         97   ctx.ensure_initialized()
    ---> 98   return ops.EagerTensor(value, ctx.device_name, dtype)
         99 
        100 


    ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float).



```
sentence1_input = tf.keras.Input(shape=(), dtype=tf.string, name = "Sentence1")
sentence2_input = tf.keras.Input(shape=(), dtype=tf.string, name = "Sentence2")
relevancyinput = tf.keras.Input(shape=(1), name = "RelevancyScore")

sentencepreprocessing = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing1')

sentence1_preprocessing = sentencepreprocessing(sentence1_input)
sentence2_preprocessing = sentencepreprocessing(sentence2_input)

sentencebert = hub.KerasLayer(tfhub_handle_encoder, trainable=False, name='BERT_encoder1')

sentence1_bert = sentencebert(sentence1_preprocessing)
sentence2_bert = sentencebert(sentence2_preprocessing)


concatenate_all = tf.keras.layers.concatenate([relevancyinput, sentence1_bert["pooled_output"], sentence2_bert["pooled_output"]])
neta2 = tf.keras.layers.Dense(1024, activation=None, name='sts1')(concatenate_all)
neta3 = tf.keras.layers.Dense(768, activation=None, name='sts2')(neta2)
net = tf.keras.layers.Dense(1, activation=None, name='sts')(neta3)

model = tf.keras.Model(
    inputs=[sentence1_input, sentence2_input],
    outputs=[net],
)
```


```
def CosineSimilarity(a,b):
  #a**b/squareroot(summation of a^2)**squaeroot(summation of b^2)
  for i in a:
    for o im b:
      a *
```


```
[print(i.shape, i.dtype) for i in model.inputs]
[print(o.shape, o.dtype) for o in model.outputs]
[print(l.name, l.input_shape, l.dtype) for l in model.layers]
model.summary()
```

    (None,) <dtype: 'string'>
    (None,) <dtype: 'string'>
    (None, 1) <dtype: 'float32'>
    Sentence1 [(None,)] string
    Sentence2 [(None,)] string
    preprocessing1 None float32
    BERT_encoder1 {'input_word_ids': (None, 128), 'input_mask': (None, 128), 'input_type_ids': (None, 128)} float32
    concatenate_5 [(None, 512), (None, 512)] float32
    sts1 (None, 1024) float32
    sts2 (None, 1024) float32
    sts (None, 768) float32
    Model: "model_5"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    Sentence1 (InputLayer)          [(None,)]            0                                            
    __________________________________________________________________________________________________
    Sentence2 (InputLayer)          [(None,)]            0                                            
    __________________________________________________________________________________________________
    preprocessing1 (KerasLayer)     {'input_word_ids': ( 0           Sentence1[0][0]                  
                                                                     Sentence2[0][0]                  
    __________________________________________________________________________________________________
    BERT_encoder1 (KerasLayer)      {'pooled_output': (N 28763649    preprocessing1[0][0]             
                                                                     preprocessing1[0][1]             
                                                                     preprocessing1[0][2]             
                                                                     preprocessing1[1][0]             
                                                                     preprocessing1[1][1]             
                                                                     preprocessing1[1][2]             
    __________________________________________________________________________________________________
    concatenate_5 (Concatenate)     (None, 1024)         0           BERT_encoder1[0][5]              
                                                                     BERT_encoder1[1][5]              
    __________________________________________________________________________________________________
    sts1 (Dense)                    (None, 1024)         1049600     concatenate_5[0][0]              
    __________________________________________________________________________________________________
    sts2 (Dense)                    (None, 768)          787200      sts1[0][0]                       
    __________________________________________________________________________________________________
    sts (Dense)                     (None, 1)            769         sts2[0][0]                       
    ==================================================================================================
    Total params: 30,601,218
    Trainable params: 1,837,569
    Non-trainable params: 28,763,649
    __________________________________________________________________________________________________



```
                                                                                                                                                                                                                      #Defining the loss function
loss = tf.keras.losses.MeanAbsoluteError()
metrics = tf.metrics.Accuracy()

#Defining the Optimizer
epochs = 4
steps_per_epoch = 5
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,num_train_steps=num_train_steps,num_warmup_steps=num_warmup_steps,optimizer_type='adamw')

#Loading the BERT Model and Training
model.compile(optimizer=optimizer,loss=loss,metrics=['mse', 'mae', 'mape', 'cosine_similarity','accuracy'])

print(df3["Relevancy_Score"])

sentence1_data = (df3.Sentence_1).astype("string")
relevancydata = pd.to_numeric((df3['Relevancy_Score']),errors='coerce')
normalized = relevancydata.div(5)
sentence2_data = (df3.Sentence_2).astype("string")
print(normalized)
print(type(sentence1_data[1]))

model.fit(
    {"Sentence1": sentence1_data, "Sentence2": sentence2_data}, y = normalized, epochs=epochs )

print(f'Training model with {tfhub_handle_encoder}')
```

    776     4.0
    1424    3.0
    227     4.7
    2402    0.8
    104     5.0
           ... 
    409     4.4
    1623    2.8
    115     5.0
    288     4.6
    2510    0.4
    Name: Relevancy_Score, Length: 2305, dtype: float64
    776     0.80
    1424    0.60
    227     0.94
    2402    0.16
    104     1.00
            ... 
    409     0.88
    1623    0.56
    115     1.00
    288     0.92
    2510    0.08
    Name: Relevancy_Score, Length: 2305, dtype: float64
    <class 'str'>
    Epoch 1/4
    73/73 [==============================] - 312s 4s/step - loss: 0.4590 - mse: 0.3090 - mae: 0.4590 - mape: 98865381.2973 - cosine_similarity: 0.9011 - accuracy: 0.0438
    Epoch 2/4
    73/73 [==============================] - 306s 4s/step - loss: 0.4504 - mse: 0.2993 - mae: 0.4504 - mape: 89167020.2162 - cosine_similarity: 0.9108 - accuracy: 0.0476
    Epoch 3/4
    73/73 [==============================] - 308s 4s/step - loss: 0.4469 - mse: 0.2979 - mae: 0.4469 - mape: 92691536.1622 - cosine_similarity: 0.9073 - accuracy: 0.0570
    Epoch 4/4
    73/73 [==============================] - 305s 4s/step - loss: 0.4534 - mse: 0.3041 - mae: 0.4534 - mape: 95444199.7838 - cosine_similarity: 0.9046 - accuracy: 0.0450
    Training model with https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1



```



df3_test = df2_test.dropna()

print(df3_test.describe().transpose())

sentence1_test = df3_test["Sentence_1"]
sentence2_test = df3_test["Sentence_2"]
relevancy_test = df3_test["Relevancy_Score"]
normalized_test = relevancy_test.div(5)
metric = tf.metrics.CosineSimilarity()
losses = model.evaluate({"Sentence1": sentence1_test, "Sentence2": sentence2_test}, {"sts": normalized_test})

print(f'Loss: {losses}')

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-9144b8188c9a> in <module>()
          2 
          3 
    ----> 4 df3_test = df2_test.dropna()
          5 
          6 print(df3_test.describe().transpose())


    NameError: name 'df2_test' is not defined



```
uploaded2 = files.upload()

predict = pd.read_csv(io.BytesIO(uploaded2['predict.csv']))


sentence1_predict = predict["Sentence_1"]
sentence2_predict = predict["Sentence_2"]
relevancy_predict = predict["Relevancy_Score"]
normalized_predict = relevancy_predict.div(5)


solutions = model.predict({"Sentence1": sentence1_predict, "Sentence2": sentence2_predict})
```



<input type="file" id="files-96231d17-f920-44cd-b706-66c27ac114d1" name="files[]" multiple disabled
   style="border:none" />
<output id="result-96231d17-f920-44cd-b706-66c27ac114d1">
 Upload widget is only available when the cell has been executed in the
 current browser session. Please rerun this cell to enable.
 </output>
 <script src="/nbextensions/google.colab/files.js"></script> 


    Saving predict.csv to predict (1).csv



```

```


```

newList = []
for x in solutions:
    newList.append(x*5)

df = pd.DataFrame(solutions)
fileName = 'prdict.csv'
df.to_csv(fileName)

print(df)
```

               0
    0  -0.296717
    1  -0.083558
    2   0.316203
    3  -0.035383
    4   0.340398
    5   0.672012
    6   0.557940
    7   0.458394
    8   0.027319
    9   0.368085
    10  0.478323
    11  0.515013
    12  1.077829
    13  0.733209
    14  0.935209
    15  0.316129
    16  0.044420
    17  0.027349
    18  0.295618
    19  0.381707
    20  0.388599
    21  0.310701
    22  0.688457
    23 -0.062880
    24  0.946695
    25 -0.148819
    26  0.647570
    27  0.811228
    28  0.595808
    29  0.022950



```
a = (df.abs())

b = normalized_predict

diff_pred = (b).subtract(a)

print(diff_pred[0].abs())

diff_pred[0].abs().describe().transpose()

print((1 - diff_pred[0].abs()).sum())
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-a4cb93cb6cf5> in <module>()
    ----> 1 a = (df.abs())
          2 
          3 b = normalized_predict
          4 
          5 diff_pred = (b).subtract(a)


    NameError: name 'df' is not defined



```
dataset_name = 'imdb'
saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))
model.save(saved_model_path, include_optimizer=False)
```

    WARNING:absl:Found untraced functions such as restored_function_body, restored_function_body, restored_function_body, restored_function_body, restored_function_body while saving (showing 5 of 310). These functions will not be directly callable after loading.
    WARNING:absl:Found untraced functions such as restored_function_body, restored_function_body, restored_function_body, restored_function_body, restored_function_body while saving (showing 5 of 310). These functions will not be directly callable after loading.

