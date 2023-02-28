Multi-label Bert
================


.. code:: ipython3

    from src.annotation.doccano import get_latest_annotated_data
    from src.data.preprocessing import DataPreprocessorHelland
    from sklearn.model_selection import train_test_split
    from transformers import AutoTokenizer, BertForSequenceClassification
    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from datasets import Dataset, DatasetDict
    import ray.data
    from ray.data.preprocessors import BatchMapper
    import pandas as pd
    from sklearn.preprocessing import MultiLabelBinarizer
    import numpy as np
    import evaluate
    from src.slack_alert.sofus_alert import sofus_alert

.. code:: ipython3

    data = get_latest_annotated_data().loc[:, ['text', 'label']].dropna().reset_index()
    data = data.loc[:, ['text', 'label']]
    data_preprocessor = DataPreprocessorHelland(data['text'])
    data_preprocessor.make_lower_cased()
    data_preprocessor.remove_extra_spaces()
    data['text'] = data_preprocessor.text_series
    one_hot = MultiLabelBinarizer()
    one_hot_label = one_hot.fit_transform(data['label'])
    one_hot_label = [list(map(float, x)) for x in one_hot_label]
    data['label'] = pd.Series(list(one_hot_label))
    data




.. raw:: html

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
          <th>text</th>
          <th>label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>premier league valgte å offentliggjøre tre fly...</td>
          <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ...</td>
        </tr>
        <tr>
          <th>1</th>
          <td>romelu lukaku har uttalt at han er misfornøyd ...</td>
          <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
        </tr>
        <tr>
          <th>2</th>
          <td>– la oss være ærlige. jeg liker det ikke, for ...</td>
          <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
        </tr>
        <tr>
          <th>3</th>
          <td>manchester united har fått mye kritikk etter å...</td>
          <td>[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
        </tr>
        <tr>
          <th>4</th>
          <td>her er torsdagens oddstips!</td>
          <td>[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, ...</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>1997</th>
          <td>marko gruljic fra red star belgrade 6. januar ...</td>
          <td>[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, ...</td>
        </tr>
        <tr>
          <th>1998</th>
          <td>tv 2s fotballkommentator øyvind alsaker mener ...</td>
          <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
        </tr>
        <tr>
          <th>1999</th>
          <td>de fire beste i premier league får spille i ch...</td>
          <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
        </tr>
        <tr>
          <th>2000</th>
          <td>– virgil van dijk, alisson becker og mohamed s...</td>
          <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
        </tr>
        <tr>
          <th>2001</th>
          <td>1. januar åpner overgangsvinduet, og storklubb...</td>
          <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
        </tr>
      </tbody>
    </table>
    <p>2002 rows × 2 columns</p>
    </div>



.. code:: ipython3

    train_split, validation_split = train_test_split(data, test_size=0.2)

.. code:: ipython3

    train_dataset = Dataset.from_pandas(train_split)
    train_dataset = train_dataset.remove_columns(['__index_level_0__'])
    validation_dataset = Dataset.from_pandas(validation_split)
    validation_dataset = validation_dataset.remove_columns(['__index_level_0__'])
    split_dict = {'train': train_dataset, 'validation': validation_dataset}
    datasets = DatasetDict(split_dict)
    datasets




.. parsed-literal::

    DatasetDict({
        train: Dataset({
            features: ['text', 'label'],
            num_rows: 1601
        })
        validation: Dataset({
            features: ['text', 'label'],
            num_rows: 401
        })
    })



.. code:: ipython3

    ray_datasets = ray.data.from_huggingface(datasets)
    ray_datasets


.. parsed-literal::

    2023-02-28 14:46:59,485	INFO worker.py:1538 -- Started a local Ray instance.
    



.. parsed-literal::

    {'train': Dataset(num_blocks=1, num_rows=1601, schema={text: string, label: list<item: double>}),
     'validation': Dataset(num_blocks=1, num_rows=401, schema={text: string, label: list<item: double>})}



.. code:: ipython3

    model_checkpoint = 'NbAiLab/nb-bert-large'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, model_max_length=512)

.. code:: ipython3

    def preprocess_function(examples):
    
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
    
    encoded_dataset = datasets.map(preprocess_function, batched=True)
    encoded_dataset



.. parsed-literal::

      0%|          | 0/2 [00:00<?, ?ba/s]



.. parsed-literal::

      0%|          | 0/1 [00:00<?, ?ba/s]




.. parsed-literal::

    DatasetDict({
        train: Dataset({
            features: ['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
            num_rows: 1601
        })
        validation: Dataset({
            features: ['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
            num_rows: 401
        })
    })



.. code:: ipython3

    metric_name = "accuracy"
    model_name = model_checkpoint.split("/")[-1]
    task = 'multi_label_classification'
    batch_size = 16
    
    args = TrainingArguments(
        f"{model_name}-finetuned-{task}",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        push_to_hub=True,
    )

.. code:: ipython3

    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
    from transformers import EvalPrediction
    import torch
        
    # source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
    def multi_label_metrics(predictions, labels, threshold=0.5):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {'f1': f1_micro_average,
                   'roc_auc': roc_auc,
                   'accuracy': accuracy}
        return metrics
    
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, 
                tuple) else p.predictions
        result = multi_label_metrics(
            predictions=preds, 
            labels=p.label_ids)
        return result

.. code:: ipython3

    model = BertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=22, problem_type="multi_label_classification")


.. parsed-literal::

    Some weights of the model checkpoint at NbAiLab/nb-bert-large were not used when initializing BertForSequenceClassification: ['cls.seq_relationship.bias', 'cls.predictions.bias', 'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.seq_relationship.weight', 'cls.predictions.decoder.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.LayerNorm.bias']
    - This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at NbAiLab/nb-bert-large and are newly initialized: ['classifier.bias', 'classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

.. code:: ipython3

    validation_key = "validation"
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )


.. parsed-literal::

    C:\Users\Eirik\sports-article-classification\src\modelling\nb-bert-large-finetuned-multi_label_classification is already a clone of https://huggingface.co/hellund/nb-bert-large-finetuned-multi_label_classification. Make sure you pull the latest changes with `repo.git_pull()`.
    

.. code:: ipython3

    trainer.train("nb-bert-large-finetuned-multi_label_classification/checkpoint-303")
    sofus_alert()


.. parsed-literal::

    Loading model from nb-bert-large-finetuned-multi_label_classification/checkpoint-303.
    The following columns in the training set don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: text. If text are not expected by `BertForSequenceClassification.forward`,  you can safely ignore this message.
    ***** Running training *****
      Num examples = 1601
      Num Epochs = 5
      Instantaneous batch size per device = 16
      Total train batch size (w. parallel, distributed & accumulation) = 16
      Gradient Accumulation steps = 1
      Total optimization steps = 505
      Number of trainable parameters = 355109910
      Continuing training from checkpoint, will skip to saved global_step
      Continuing training from epoch 3
      Continuing training from global step 303
      Will skip the first 3 epochs then the first 0 batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` flag to your launch command, but you will resume the training on data already seen by your model.
    


.. parsed-literal::

    0it [00:00, ?it/s]



.. raw:: html

    
        <div>
    
          <progress value='505' max='505' style='width:300px; height:20px; vertical-align: middle;'></progress>
          [505/505 1:36:48, Epoch 5/5]
        </div>
        <table border="1" class="dataframe">
      <thead>
     <tr style="text-align: left;">
          <th>Epoch</th>
          <th>Training Loss</th>
          <th>Validation Loss</th>
          <th>F1</th>
          <th>Roc Auc</th>
          <th>Accuracy</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>4</td>
          <td>No log</td>
          <td>0.136843</td>
          <td>0.635762</td>
          <td>0.756853</td>
          <td>0.426434</td>
        </tr>
        <tr>
          <td>5</td>
          <td>0.135300</td>
          <td>0.134019</td>
          <td>0.647781</td>
          <td>0.762611</td>
          <td>0.428928</td>
        </tr>
      </tbody>
    </table><p>


.. parsed-literal::

    The following columns in the evaluation set don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: text. If text are not expected by `BertForSequenceClassification.forward`,  you can safely ignore this message.
    ***** Running Evaluation *****
      Num examples = 401
      Batch size = 16
    Saving model checkpoint to nb-bert-large-finetuned-multi_label_classification\checkpoint-404
    Configuration saved in nb-bert-large-finetuned-multi_label_classification\checkpoint-404\config.json
    Model weights saved in nb-bert-large-finetuned-multi_label_classification\checkpoint-404\pytorch_model.bin
    tokenizer config file saved in nb-bert-large-finetuned-multi_label_classification\checkpoint-404\tokenizer_config.json
    Special tokens file saved in nb-bert-large-finetuned-multi_label_classification\checkpoint-404\special_tokens_map.json
    tokenizer config file saved in nb-bert-large-finetuned-multi_label_classification\tokenizer_config.json
    Special tokens file saved in nb-bert-large-finetuned-multi_label_classification\special_tokens_map.json
    The following columns in the evaluation set don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: text. If text are not expected by `BertForSequenceClassification.forward`,  you can safely ignore this message.
    ***** Running Evaluation *****
      Num examples = 401
      Batch size = 16
    Saving model checkpoint to nb-bert-large-finetuned-multi_label_classification\checkpoint-505
    Configuration saved in nb-bert-large-finetuned-multi_label_classification\checkpoint-505\config.json
    Model weights saved in nb-bert-large-finetuned-multi_label_classification\checkpoint-505\pytorch_model.bin
    tokenizer config file saved in nb-bert-large-finetuned-multi_label_classification\checkpoint-505\tokenizer_config.json
    Special tokens file saved in nb-bert-large-finetuned-multi_label_classification\checkpoint-505\special_tokens_map.json
    tokenizer config file saved in nb-bert-large-finetuned-multi_label_classification\tokenizer_config.json
    Special tokens file saved in nb-bert-large-finetuned-multi_label_classification\special_tokens_map.json
    
    
    Training completed. Do not forget to share your model on huggingface.co/models =)
    
    
    Loading best model from nb-bert-large-finetuned-multi_label_classification\checkpoint-505 (score: 0.428927680798005).
    

.. parsed-literal::

    Sofus has sent an alert - Check slack!
    

.. code:: ipython3

    trainer.evaluate()


.. parsed-literal::

    The following columns in the evaluation set don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: text. If text are not expected by `BertForSequenceClassification.forward`,  you can safely ignore this message.
    ***** Running Evaluation *****
      Num examples = 401
      Batch size = 16
    


.. raw:: html

    
    <div>
    
      <progress value='26' max='26' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [26/26 03:27]
    </div>
    




.. parsed-literal::

    {'eval_loss': 0.13401861488819122,
     'eval_f1': 0.647780925401322,
     'eval_roc_auc': 0.7626107290024299,
     'eval_accuracy': 0.428927680798005,
     'eval_runtime': 215.6616,
     'eval_samples_per_second': 1.859,
     'eval_steps_per_second': 0.121,
     'epoch': 5.0}



.. code:: ipython3

    text = '- Jeg elsker å score mål på corner. Vi var mye bedre i luften enn Brann idag'
    encoding = tokenizer(text, return_tensors="pt")
    
    outputs = trainer.model(**encoding)

.. code:: ipython3

    logits = outputs.logits

.. code:: ipython3

    # apply sigmoid + threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    # turn predicted id's into actual label names
    predictions = np.reshape(predictions, (1, predictions.shape[0]))
    predicted_labels = one_hot.inverse_transform(predictions)
    print(predicted_labels)


.. parsed-literal::

    [('Quote',)]
    
