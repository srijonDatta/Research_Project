# Research_Project

**Extraction of the Triggering Causes of a Query Event** - The prime objective of this work is to investigate that <b>'given a causal query (rather any effect mentioned in the query), how well a supervised retrieval set up can enumerate the list of plausible triggering causes embedded in the documents?'</b> 

<b>Abstruct:</b> The main goal of traditional information retrieval systems is to find documents that are pertinent to a certain query idea. But when working with sources like collections of news articles, a user may frequently want to find documents that explain the series of circumstances that may have led to the news event in addition to those that describe the news event itself. Because they involve several underlying causative components, these interactions may be intricate. In response to this demand from the issue, we create the aim of causal information extraction. This work uses a Convolutional Neural Network (CNN) and a Transformer-based model to give an in-depth structure for causality-driven document classification. The overall architecture includes phases for gathering data, extracting information from documents, indexing, and creating input vectors for models. Regarding causal queries, the suggested models successfully separate relevant and irrelevant content. The Transformer-based BERT model outperforms all others in experimental assessment, effectively predicting document relevance with nearly 72% accuracy rate. The work demonstrates the potential of data-driven models in addressing difficult information retrieval problems and identifies prospective directions for model optimization and dataset augmentation in the future.

<b>Key Findings:</b> Both the proposed CNN and transformer-based models leverage the advantage of classi- fication. The model’s efficacy is then measured in terms of ‘accuracy’ when it performs as a binary classifier and predicts which of any given two input documents is relevant. Thus, we measure the % of total correctly predicted documents per query compared to the ground truth. Experimental results show that CNN-based model achieved nearly 66% accuracy, where, transformer-based model outperformed the former one with nearly 72% accurate predictions.

<b>Important Instructions to Regenerate the Work :</b>

1. **Data Acccess -** To understand the structure of the data corpus i.e. the whole collection that has been used in this work, please check the project report file named as 'MSc_Research_Project_Report_21225265.pdf' (page 5). To protect the Data Privacy complience, the raw data has not been made public but if you wish to use so please connect me via email (srijondatta21@gmail.com) with valid reasons and purpose of useage or can visit this [WEBSITE](http://fire.irsi.res.in/fire/static/data) and request the access key to the authority. Similarly to use the 'Topic Set' and the 'Relevance Judgement Set' please check the citations mentioned on the main report file (MSc_Research_Project_Report_21225265.pdf, page 6).

2. **Code Files -** To access all the code files, please download the 'Code_Artifacts.zip'. To get a high-level view of the code files along with the codes, please follow the step-by-step guidence, documented on 'MSc_Research_Project_Configuration_Manual_21225265'. A full list of pre-requisites has been given in the main report file (page 9-10).

3. *Model-Architectures* --------

3.1. CNN Architecture: 
![CNN Architecture](https://github.com/srijonDatta/Research_Project/blob/main/Model_Architectures/CNN_Architecture.png)

3.1. BERT Architecture: 
![BERT Architecture](https://github.com/srijonDatta/Research_Project/blob/main/Model_Architectures/BERT_Architecture.png)


## NOTE: 
<b>[MSc_Research_Project_Report_21225265.pdf](https://github.com/srijonDatta/Research_Project/blob/main/MSc_Research_Project_Report_21225265.pdf) is the complete documentation of the entire reseach work and to get a clear idea about this investigation, I would like to request all the current viewer to go through this report file first.</b>  

