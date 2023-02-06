# sports-article-classification
Full project of master thesis containing data collection and machine learning of text from sports articles.


## Project Description
This project uses a collection of python scripts and notebooks to apply machine learning to the text classification problem of correctly labeling pragraphs from football articles. The project utilzes different techniques such as web scraping, clustering, active learning and doccano to create and annotate a new dataset. The dataset contains all articles written and published on VG.no with the tag football in 2022. The ultimate goal of the project is to showcase the use of Norwegian trained BERT-models on a Norwegian text classifcation problem. 

## Doccano
Docanno is used to annotate the text data with ease. It supports multi label annotation and active learning. Active learning can help annotate a large dataset, but must be used with caution and tight supervision.
![Doccano example](https://raw.githubusercontent.com/hellund/sports-article-classification/main/images/doccano_example.png)


## Labels
The decision of which labels to include is tough. This hierarchy helps displaying the though process of which labels are to be inbcluded in the annotation of the dataset. The goal is to be able to showcase the versatility of transformer models. Labels from different levels of information have been chosen. In this case from level 2, 3 and 4 as shown in the hiearchy below. Since "Game action" can be aggregated from level 4, it has not been included in the annotation. 
![Labeling hierarchy](https://raw.githubusercontent.com/hellund/sports-article-classification/main/images/label_hierarchy.drawio.png)

