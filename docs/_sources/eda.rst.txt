EDA
=======

Imports
~~~~~~~

.. code:: ipython3

    import pandas as pd
    import re
    import string
    import plotly.express as px
    import plotly.figure_factory as ff
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    from nltk.probability import FreqDist
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    
    from src.data.nordskog_data import get_data
    from src.data.preprocessing import DataPreprocessor

Loading data
~~~~~~~~~~~~

.. code:: ipython3

    train, test = get_data()

About the data
~~~~~~~~~~~~~~

The data is a collection of fotball articles collected by Aanund
Nordskog from VG and TV2, divided into paragraphs. The training data was
retrieved form articles published between 22.07.18 and 17.09.18. The
testing data is from articles between 14.12.19. Not that the the
different time periods may affect the kind of articles published in the
different periods. Each paragraph is labeled as following:

.. raw:: html

   <ul>

.. raw:: html

   <li>

Goal/Assist

.. raw:: html

   </li>

.. raw:: html

   <li>

Quotes

.. raw:: html

   </li>

.. raw:: html

   <li>

Transfer

.. raw:: html

   </li>

.. raw:: html

   <li>

Irrelevant

.. raw:: html

   </li>

.. raw:: html

   <li>

Ignore

.. raw:: html

   </li>

.. raw:: html

   <li>

Player Detail

.. raw:: html

   </li>

.. raw:: html

   <li>

Club Detail

.. raw:: html

   </li>

.. raw:: html

   <li>

Chances

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. code:: ipython3

    # Printing the dimension of the data
    print(f'Number of training paragraphs:{train.shape}')
    print(f'Number of testing paragraphs: {test.shape}')


.. parsed-literal::

    Number of training paragraphs:(5526, 2)
    Number of testing paragraphs: (611, 2)
    

.. code:: ipython3

    # Displaying the first rows of training data
    train.head()




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
          <td>Vålerenga - Rosenborg 2-3</td>
          <td>Ignore</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Sam Johnson ga vertene ledelsen, men Jonathan ...</td>
          <td>Goal/Assist</td>
        </tr>
        <tr>
          <th>2</th>
          <td>På et hjørnespark langt på overtid kom avgjøre...</td>
          <td>Goal/Assist</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Ti minutter før pause scoret Sam Johnson sitt ...</td>
          <td>Goal/Assist</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Vålerenga holdt 1-0-ledelsen bare frem til sis...</td>
          <td>Goal/Assist</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    test.head()




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
          <td>Se Manchester City-Chelsea søndag fra 16.30 (k...</td>
          <td>Ignore</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Chelsea har en lang tradisjon for å hente stje...</td>
          <td>Irrelevant</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Spillere som Andrij Sjevtsjenko, Fernando Torr...</td>
          <td>Irrelevant</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Blant alle skuffelsene har det likevel vært ly...</td>
          <td>Irrelevant</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Det er denne rekken Higuaín nå håper å føye se...</td>
          <td>Irrelevant</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    # Displaying info
    print('Training data:')
    print(train.info())
    print('\nTesting data:')
    print(test.info())


.. parsed-literal::

    Training data:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5526 entries, 0 to 5525
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   text    5526 non-null   object
     1   label   5526 non-null   object
    dtypes: object(2)
    memory usage: 86.5+ KB
    None
    
    Testing data:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 611 entries, 0 to 610
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   text    611 non-null    object
     1   label   611 non-null    object
    dtypes: object(2)
    memory usage: 9.7+ KB
    None
    

.. code:: ipython3

    # Display statistical information for categorical variables
    train.describe(include=object)




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
          <th>count</th>
          <td>5526</td>
          <td>5526</td>
        </tr>
        <tr>
          <th>unique</th>
          <td>5436</td>
          <td>12</td>
        </tr>
        <tr>
          <th>top</th>
          <td>Tipster gir deg ferske oddstips hver dag!</td>
          <td>Goal/Assist</td>
        </tr>
        <tr>
          <th>freq</th>
          <td>11</td>
          <td>1117</td>
        </tr>
      </tbody>
    </table>
    </div>



There are 90 paragraphs that are not unique, with “Tipster gir deg…”
being the most frequent.

.. code:: ipython3

    train[train['text'].duplicated()]




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
          <th>83</th>
          <td>Se målene i Sportsnyhetene øverst!\n</td>
          <td>Ignore</td>
        </tr>
        <tr>
          <th>378</th>
          <td>Se målene i Sportsnyhetene øverst!\n</td>
          <td>Ignore</td>
        </tr>
        <tr>
          <th>840</th>
          <td>Tipster gir deg ferske oddstips hver dag!</td>
          <td>Ignore</td>
        </tr>
        <tr>
          <th>846</th>
          <td>Tipster gir deg ferske oddstips hver dag!</td>
          <td>Ignore</td>
        </tr>
        <tr>
          <th>847</th>
          <td>• Her er fredagens oddstips</td>
          <td>Ignore</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>5376</th>
          <td>Se generaltabben i videovinduet øverst.</td>
          <td>Ignore</td>
        </tr>
        <tr>
          <th>5383</th>
          <td>Se Leicester mot Liverpool på TV 2 Sport Premi...</td>
          <td>Ignore</td>
        </tr>
        <tr>
          <th>5387</th>
          <td>Se Europaligaen på TV 2 Sumo og TV 2s kanaler!\n</td>
          <td>Ignore</td>
        </tr>
        <tr>
          <th>5398</th>
          <td>VG Live: Følg Norge-Nederland kl. 17.00</td>
          <td>Ignore</td>
        </tr>
        <tr>
          <th>5400</th>
          <td>Norge-Kypros ser du på TV 2 eller følger på V...</td>
          <td>Ignore</td>
        </tr>
      </tbody>
    </table>
    <p>90 rows × 2 columns</p>
    </div>



.. code:: ipython3

    # Display statistical information for categorical variables
    test.describe(include=object)




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
          <th>count</th>
          <td>611</td>
          <td>611</td>
        </tr>
        <tr>
          <th>unique</th>
          <td>609</td>
          <td>5</td>
        </tr>
        <tr>
          <th>top</th>
          <td>– Selvfølgelig vil jeg tilbake til Madrid og s...</td>
          <td>Irrelevant</td>
        </tr>
        <tr>
          <th>freq</th>
          <td>2</td>
          <td>293</td>
        </tr>
      </tbody>
    </table>
    </div>



There are at least two duplicate paragraphs in the testing data

.. code:: ipython3

    test[test['text'].duplicated()]




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
          <th>290</th>
          <td>– Selvfølgelig vil jeg tilbake til Madrid og s...</td>
          <td>Quote</td>
        </tr>
        <tr>
          <th>588</th>
          <td>– Jeg har mange drømmer, også her i Vitesse. Å...</td>
          <td>Quote</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    train['label'].value_counts()




.. parsed-literal::

    Goal/Assist       1117
    quote              975
    Transfer           887
    irrelevant         812
    Ignore             663
    Player details     340
    Club details       315
    sjanse             300
    Injuries            59
    Rodt/gult kort      50
    Club drama           5
    Personal drama       3
    Name: label, dtype: int64



.. code:: ipython3

    test['label'].value_counts()




.. parsed-literal::

    Irrelevant     293
    Quote          141
    Goal/Assist     69
    Transfer        60
    Ignore          48
    Name: label, dtype: int64



Data preprocessing
~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    processed_train = DataPreprocessor(train)
    processed_train.map_nordskog_data()
    train = processed_train.data.copy()

.. code:: ipython3

    train['num_char'] = train['text'].map(lambda x: len(x))
    train['num_words'] = train['text'].map(lambda x: len(x.split()))
    train['num_sent'] = train['text'].map(lambda x: len(sent_tokenize(x)))
    train['avg_word_len'] = train['text'].map(lambda x: np.mean([len(word) for word in str(x).split()]))
    train['avg_sent_len'] = train['text'].map(lambda x: np.mean([len(word.split()) for word in sent_tokenize(x)]))
    train




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
          <th>num_char</th>
          <th>num_words</th>
          <th>num_sent</th>
          <th>avg_word_len</th>
          <th>avg_sent_len</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Vålerenga - Rosenborg 2-3</td>
          <td>Ignore</td>
          <td>25</td>
          <td>4</td>
          <td>1</td>
          <td>5.500000</td>
          <td>4.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Sam Johnson ga vertene ledelsen, men Jonathan ...</td>
          <td>Goal/Assist</td>
          <td>187</td>
          <td>31</td>
          <td>2</td>
          <td>5.064516</td>
          <td>15.5</td>
        </tr>
        <tr>
          <th>2</th>
          <td>På et hjørnespark langt på overtid kom avgjøre...</td>
          <td>Goal/Assist</td>
          <td>134</td>
          <td>20</td>
          <td>1</td>
          <td>5.750000</td>
          <td>20.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Ti minutter før pause scoret Sam Johnson sitt ...</td>
          <td>Goal/Assist</td>
          <td>166</td>
          <td>31</td>
          <td>1</td>
          <td>4.387097</td>
          <td>31.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Vålerenga holdt 1-0-ledelsen bare frem til sis...</td>
          <td>Goal/Assist</td>
          <td>203</td>
          <td>29</td>
          <td>2</td>
          <td>6.034483</td>
          <td>14.5</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>5521</th>
          <td>– Mateo har sagt at han ønsker å dra. Jeg vil ...</td>
          <td>Quote</td>
          <td>152</td>
          <td>29</td>
          <td>2</td>
          <td>4.275862</td>
          <td>14.5</td>
        </tr>
        <tr>
          <th>5522</th>
          <td>– Her gjør han en miss. Han står midt i mål, o...</td>
          <td>Quote</td>
          <td>242</td>
          <td>46</td>
          <td>4</td>
          <td>4.282609</td>
          <td>11.5</td>
        </tr>
        <tr>
          <th>5523</th>
          <td>– Vi kan ta med masse positivt fra kampen, for...</td>
          <td>Quote</td>
          <td>213</td>
          <td>42</td>
          <td>2</td>
          <td>4.095238</td>
          <td>21.0</td>
        </tr>
        <tr>
          <th>5524</th>
          <td>Den tyske midtbanespilleren kom til Bayern Mün...</td>
          <td>Player Detail</td>
          <td>327</td>
          <td>53</td>
          <td>2</td>
          <td>4.735849</td>
          <td>26.5</td>
        </tr>
        <tr>
          <th>5525</th>
          <td>Bendtner har vært i norsk fotball siden mars 2...</td>
          <td>Player Detail</td>
          <td>170</td>
          <td>29</td>
          <td>2</td>
          <td>4.896552</td>
          <td>14.5</td>
        </tr>
      </tbody>
    </table>
    <p>5526 rows × 7 columns</p>
    </div>



.. code:: ipython3

    px.bar(train.groupby(['label'], as_index=False).agg(count=('label', 'count')),
                 x='label', y='count',
                 color='label',
                 labels={'label': 'Labels', 'count': 'Amount of paragraphs'},
                 title='Sum of paragraphs per label in training data')


.. image:: output_21_1.png
.. code:: ipython3

    px.bar(train.groupby(['label'], as_index=False).agg(sum_words=('num_words', 'sum')),
                 x='label', y='sum_words',
                 color='label',
                 labels={'label': 'Labels', 'sum_words': 'Amount of words'},
                 title='Sum of words per label in training data')


.. image:: output_22_0.png
.. code:: ipython3

    px.bar(train.groupby(['label'], as_index=False).agg(sum_char=('num_char', 'sum')),
                 x='label', y='sum_char',
                 color='label',
                 labels={'label': 'Labels', 'sum_char': 'Amount of characters'},
                 title='Sum of characters per label in training data')


.. image:: output_23_0.png
.. code:: ipython3

    px.bar(train.groupby(['label'], as_index=False).agg(sum_sent=('num_sent', 'sum')),
                 x='label', y='sum_sent',
                 color='label',
                 labels={'label': 'Labels', 'sum_sent': 'Amount of characters'},
                 title='Sum of sentences per label in training data')


.. image:: output_24_0.png
.. code:: ipython3

    plots = train.columns[2:].to_list()
    
    for plot in plots:
        fig = ff.create_distplot([train[plot]], [plot], curve_type='kde')
        fig.update_layout(title = plot.capitalize())
        fig.show()



.. image:: output_25_0.png
.. image:: output_25_1.png
.. image:: output_25_2.png
.. image:: output_25_3.png
.. image:: output_25_4.png
.. code:: ipython3

    processed_train.remove_extra_spaces_from_text()
    processed_train.remove_paragraphs_over_65_words()
    train = processed_train.data.copy()

.. code:: ipython3

    stop = set(stopwords.words('norwegian'))
    corpus = [word for i in train['text'].str.split().values.tolist() for word in i if (word not in stop)]

.. code:: ipython3

    most_frequent = FreqDist(corpus).most_common(10)
    most_frequent




.. parsed-literal::

    [('–', 1173),
     ('Det', 830),
     ('fikk', 496),
     ('ballen', 476),
     ('minutter', 469),
     ('sier', 434),
     ('TV', 423),
     ('Han', 395),
     ('første', 391),
     ('to', 377)]



.. code:: ipython3

    words, frequency = [], []
    for word, count in most_frequent:
        words.append(word)
        frequency.append(count)
        
    fig = px.bar(x=frequency, y=words)
    fig.update_yaxes(autorange="reversed")



.. image:: output_29_0.png
.. code:: ipython3

    vec = CountVectorizer(stop_words = stop, ngram_range = (2,2))
    bow = vec.fit_transform(train['text'])
    count_values = bow.toarray().sum(axis=0)
    ngram_freq = pd.DataFrame(sorted([(count_values[i], k) for k, i in vec.vocabulary_.items()], reverse = True))
    ngram_freq.columns = ["frequency", "ngram"]
    ngram_freq.head(10)




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
          <th>frequency</th>
          <th>ngram</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>236</td>
          <td>premier league</td>
        </tr>
        <tr>
          <th>1</th>
          <td>153</td>
          <td>manchester united</td>
        </tr>
        <tr>
          <th>2</th>
          <td>142</td>
          <td>real madrid</td>
        </tr>
        <tr>
          <th>3</th>
          <td>118</td>
          <td>tv 2s</td>
        </tr>
        <tr>
          <th>4</th>
          <td>112</td>
          <td>millioner kroner</td>
        </tr>
        <tr>
          <th>5</th>
          <td>85</td>
          <td>tv sport</td>
        </tr>
        <tr>
          <th>6</th>
          <td>77</td>
          <td>sport premium</td>
        </tr>
        <tr>
          <th>7</th>
          <td>77</td>
          <td>minutter senere</td>
        </tr>
        <tr>
          <th>8</th>
          <td>75</td>
          <td>manchester city</td>
        </tr>
        <tr>
          <th>9</th>
          <td>68</td>
          <td>tv sumo</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    vec = CountVectorizer(stop_words = stop, ngram_range = (3,3))
    bow = vec.fit_transform(train['text'])
    count_values = bow.toarray().sum(axis=0)
    ngram_freq = pd.DataFrame(sorted([(count_values[i], k) for k, i in vec.vocabulary_.items()], reverse = True))
    ngram_freq.columns = ["frequency", "ngram"]
    ngram_freq.head(10)




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
          <th>frequency</th>
          <th>ngram</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>77</td>
          <td>tv sport premium</td>
        </tr>
        <tr>
          <th>1</th>
          <td>34</td>
          <td>erling braut håland</td>
        </tr>
        <tr>
          <th>2</th>
          <td>32</td>
          <td>tv 2s fotballekspert</td>
        </tr>
        <tr>
          <th>3</th>
          <td>32</td>
          <td>sier tv 2s</td>
        </tr>
        <tr>
          <th>4</th>
          <td>32</td>
          <td>minutter full tid</td>
        </tr>
        <tr>
          <th>5</th>
          <td>31</td>
          <td>sport premium sumo</td>
        </tr>
        <tr>
          <th>6</th>
          <td>27</td>
          <td>tv sumo tv</td>
        </tr>
        <tr>
          <th>7</th>
          <td>26</td>
          <td>2s premier league</td>
        </tr>
        <tr>
          <th>8</th>
          <td>25</td>
          <td>tv 2s premier</td>
        </tr>
        <tr>
          <th>9</th>
          <td>25</td>
          <td>premium tv sumo</td>
        </tr>
      </tbody>
    </table>
    </div>


