---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.7
  nbformat: 4
  nbformat_minor: 5
---

::: {#2473e218-2e2c-4734-a3cf-05aa9354240f .cell .markdown}
### Imports
:::

::: {#bfac88b8-ef21-4f4e-957e-36422c5c1fa8 .cell .code execution_count="1"}
``` python
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
```
:::

::: {#fddf3c8b-5452-4f98-abd1-5319412bfc03 .cell .markdown}
### Loading data
:::

::: {#89a5a4d7-88ae-4f07-91a6-fe386be34a0c .cell .code execution_count="2"}
``` python
train, test = get_data()
```
:::

::: {#88c416b4-5dce-442f-8869-cb3a946148c0 .cell .markdown}
### About the data
:::

::: {#8ce2b0c5-578f-4b04-a3e6-0aa50347d9d5 .cell .markdown}
The data is a collection of fotball articles collected by Aanund
Nordskog from VG and TV2, divided into paragraphs. The training data was
retrieved form articles published between 22.07.18 and 17.09.18. The
testing data is from articles between 14.12.19. Not that the the
different time periods may affect the kind of articles published in the
different periods. Each paragraph is labeled as following: `<ul>`{=html}
`<li>`{=html}Goal/Assist`</li>`{=html} `<li>`{=html}Quotes`</li>`{=html}
`<li>`{=html}Transfer`</li>`{=html}
`<li>`{=html}Irrelevant`</li>`{=html} `<li>`{=html}Ignore`</li>`{=html}
`<li>`{=html}Player Detail`</li>`{=html} `<li>`{=html}Club
Detail`</li>`{=html} `<li>`{=html}Chances`</li>`{=html}`</ul>`{=html}
:::

::: {#0b451d7f-9a76-40f7-9ec0-34d13eb40012 .cell .code execution_count="3"}
``` python
# Printing the dimension of the data
print(f'Number of training paragraphs:{train.shape}')
print(f'Number of testing paragraphs: {test.shape}')
```

::: {.output .stream .stdout}
    Number of training paragraphs:(5526, 2)
    Number of testing paragraphs: (611, 2)
:::
:::

::: {#b78402dc-63b8-45f6-adc5-8b3aec6932b7 .cell .code execution_count="4"}
``` python
# Displaying the first rows of training data
train.head()
```

::: {.output .execute_result execution_count="4"}
```{=html}
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
```
:::
:::

::: {#5048d331-2c6d-4d90-8c9f-377609823f0d .cell .code execution_count="5"}
``` python
test.head()
```

::: {.output .execute_result execution_count="5"}
```{=html}
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
```
:::
:::

::: {#ac62d672-1a68-46e9-85d1-f3903beddc4d .cell .code execution_count="6"}
``` python
# Displaying info
print('Training data:')
print(train.info())
print('\nTesting data:')
print(test.info())
```

::: {.output .stream .stdout}
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
:::
:::

::: {#2db82237-4194-47ea-bd57-242e3912dcae .cell .code execution_count="7"}
``` python
# Display statistical information for categorical variables
train.describe(include=object)
```

::: {.output .execute_result execution_count="7"}
```{=html}
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
```
:::
:::

::: {#9147bd52-0e87-4f30-841f-f1e43c31ced3 .cell .markdown}
There are 90 paragraphs that are not unique, with "Tipster gir deg..."
being the most frequent.
:::

::: {#cfdb6df4-e37f-4633-8122-956685ae7661 .cell .code execution_count="8"}
``` python
train[train['text'].duplicated()]
```

::: {.output .execute_result execution_count="8"}
```{=html}
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
```
:::
:::

::: {#6c195253-8535-4073-b4cf-66dd9b9c31f3 .cell .code execution_count="9"}
``` python
# Display statistical information for categorical variables
test.describe(include=object)
```

::: {.output .execute_result execution_count="9"}
```{=html}
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
```
:::
:::

::: {#62c016b9-1e22-4289-a848-c61d33a7374b .cell .markdown}
There are at least two duplicate paragraphs in the testing data
:::

::: {#68c71ef1-5e47-48cd-88be-6d0fcd84b112 .cell .code execution_count="10"}
``` python
test[test['text'].duplicated()]
```

::: {.output .execute_result execution_count="10"}
```{=html}
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
```
:::
:::

::: {#1d23f692-4c6d-4114-ac2c-db6be2ae2d61 .cell .code execution_count="11"}
``` python
train['label'].value_counts()
```

::: {.output .execute_result execution_count="11"}
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
:::
:::

::: {#83e52059-58cf-46f2-9cfb-31aca9f3cd50 .cell .code execution_count="12"}
``` python
test['label'].value_counts()
```

::: {.output .execute_result execution_count="12"}
    Irrelevant     293
    Quote          141
    Goal/Assist     69
    Transfer        60
    Ignore          48
    Name: label, dtype: int64
:::
:::

::: {#7233f497-f91a-40db-8726-ef433c8cadff .cell .markdown}
### Data preprocessing
:::

::: {#72e72561-7366-41bf-88b5-cd9b8acea053 .cell .code execution_count="13"}
``` python
processed_train = DataPreprocessor(train)
processed_train.map_nordskog_data()
train = processed_train.data.copy()
```
:::

::: {#1e6c7efc-5275-46f3-bc1a-f60fb3a6a46e .cell .code execution_count="14"}
``` python
train['num_char'] = train['text'].map(lambda x: len(x))
train['num_words'] = train['text'].map(lambda x: len(x.split()))
train['num_sent'] = train['text'].map(lambda x: len(sent_tokenize(x)))
train['avg_word_len'] = train['text'].map(lambda x: np.mean([len(word) for word in str(x).split()]))
train['avg_sent_len'] = train['text'].map(lambda x: np.mean([len(word.split()) for word in sent_tokenize(x)]))
train
```

::: {.output .execute_result execution_count="14"}
```{=html}
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
```
:::
:::

::: {#4806c357-b3bc-4dbe-95fe-4ee82556e757 .cell .code execution_count="15"}
``` python
px.bar(train.groupby(['label'], as_index=False).agg(count=('label', 'count')),
             x='label', y='count',
             color='label',
             labels={'label': 'Labels', 'count': 'Amount of paragraphs'},
             title='Sum of paragraphs per label in training data')
```

::: {.output .display_data}
```{=html}
        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        define('plotly', function(require, exports, module) {
            /**
* plotly.js v2.18.0
* Copyright 2012-2023, Plotly, Inc.
* All rights reserved.
* Licensed under the MIT license
*/
/*! For license information please see plotly.min.js.LICENSE.txt */
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        
```
:::

::: {.output .display_data}
![](vertopal_a786fffdc1ac4ea283c69c621ac3d33d/f89af6a5be7644569f3412b6c9076ee386d64db6.png)
:::
:::

::: {#de2c9917-dd4d-4c3c-b0b3-6a8d86638b8e .cell .code execution_count="16"}
``` python
px.bar(train.groupby(['label'], as_index=False).agg(sum_words=('num_words', 'sum')),
             x='label', y='sum_words',
             color='label',
             labels={'label': 'Labels', 'sum_words': 'Amount of words'},
             title='Sum of words per label in training data')
```

::: {.output .display_data}
![](vertopal_a786fffdc1ac4ea283c69c621ac3d33d/4ea409a6775dc1e4d59eaf4c24816146c3912e84.png)
:::
:::

::: {#018ca9ff-1d9f-4562-95e9-8111f34d0dfa .cell .code execution_count="17"}
``` python
px.bar(train.groupby(['label'], as_index=False).agg(sum_char=('num_char', 'sum')),
             x='label', y='sum_char',
             color='label',
             labels={'label': 'Labels', 'sum_char': 'Amount of characters'},
             title='Sum of characters per label in training data')
```

::: {.output .display_data}
![](vertopal_a786fffdc1ac4ea283c69c621ac3d33d/07132fca99affda09c54e2abba2420568c0e31e9.png)
:::
:::

::: {#27da6cbe-7582-415f-902d-075d514a68c3 .cell .code execution_count="18"}
``` python
px.bar(train.groupby(['label'], as_index=False).agg(sum_sent=('num_sent', 'sum')),
             x='label', y='sum_sent',
             color='label',
             labels={'label': 'Labels', 'sum_sent': 'Amount of characters'},
             title='Sum of sentences per label in training data')
```

::: {.output .display_data}
![](vertopal_a786fffdc1ac4ea283c69c621ac3d33d/5a71758c83e60ef2153ea0ea4653423053319c8e.png)
:::
:::

::: {#e9229b68-3a8a-4259-881c-818384db94f2 .cell .code execution_count="19"}
``` python
plots = train.columns[2:].to_list()

for plot in plots:
    fig = ff.create_distplot([train[plot]], [plot], curve_type='kde')
    fig.update_layout(title = plot.capitalize())
    fig.show()
```

::: {.output .display_data}
![](vertopal_a786fffdc1ac4ea283c69c621ac3d33d/e55b258251c0ce1329ba60cfd5353252cc838dab.png)
:::

::: {.output .display_data}
![](vertopal_a786fffdc1ac4ea283c69c621ac3d33d/9e93333be626a5740cb7143c52687df2602d7137.png)
:::

::: {.output .display_data}
![](vertopal_a786fffdc1ac4ea283c69c621ac3d33d/35c62e824486e1a7a4d4bfc50405a5513a662f01.png)
:::

::: {.output .display_data}
![](vertopal_a786fffdc1ac4ea283c69c621ac3d33d/ba1704d83b4cd14d5cc3395347b35c399e88a18f.png)
:::

::: {.output .display_data}
![](vertopal_a786fffdc1ac4ea283c69c621ac3d33d/112babe27acc9465e77fa9774c0d5e1ce9c32f4b.png)
:::
:::

::: {#3b90612d-1e2a-4b98-a770-ca528f8f3c70 .cell .code execution_count="20"}
``` python
processed_train.remove_extra_spaces_from_text()
processed_train.remove_paragraphs_over_65_words()
train = processed_train.data.copy()
```
:::

::: {#05f39e42-1275-4f7d-a37f-cc42105da090 .cell .code execution_count="21"}
``` python
stop = set(stopwords.words('norwegian'))
corpus = [word for i in train['text'].str.split().values.tolist() for word in i if (word not in stop)]
```
:::

::: {#c9dc8e34-ab31-401a-9f0b-4805c35e2a27 .cell .code execution_count="22"}
``` python
most_frequent = FreqDist(corpus).most_common(10)
most_frequent
```

::: {.output .execute_result execution_count="22"}
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
:::
:::

::: {#0868b328-eb66-4a9f-a735-cd42efc3e948 .cell .code execution_count="23"}
``` python
words, frequency = [], []
for word, count in most_frequent:
    words.append(word)
    frequency.append(count)
    
fig = px.bar(x=frequency, y=words)
fig.update_yaxes(autorange="reversed")
```

::: {.output .display_data}
![](vertopal_a786fffdc1ac4ea283c69c621ac3d33d/5451064fe4033cbe868d3159ba4c7fe24fb5ab21.png)
:::
:::

::: {#6bfe6bf0-5aac-4b9b-8a9e-413055919860 .cell .code execution_count="24"}
``` python
vec = CountVectorizer(stop_words = stop, ngram_range = (2,2))
bow = vec.fit_transform(train['text'])
count_values = bow.toarray().sum(axis=0)
ngram_freq = pd.DataFrame(sorted([(count_values[i], k) for k, i in vec.vocabulary_.items()], reverse = True))
ngram_freq.columns = ["frequency", "ngram"]
ngram_freq.head(10)
```

::: {.output .execute_result execution_count="24"}
```{=html}
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
```
:::
:::

::: {#f0e79ded-fdc3-48a3-9477-9b28e07831ca .cell .code execution_count="25"}
``` python
vec = CountVectorizer(stop_words = stop, ngram_range = (3,3))
bow = vec.fit_transform(train['text'])
count_values = bow.toarray().sum(axis=0)
ngram_freq = pd.DataFrame(sorted([(count_values[i], k) for k, i in vec.vocabulary_.items()], reverse = True))
ngram_freq.columns = ["frequency", "ngram"]
ngram_freq.head(10)
```

::: {.output .execute_result execution_count="25"}
```{=html}
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
```
:::
:::