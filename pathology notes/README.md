## Hierarchical attention networks for information extraction from cancer pathology reports

As an undergraduate, I worked for two semesters (Fall 2021-Spring 2022) on this project as a research intern with the University of Miami's Institute for Data Science and Computing.

Our goal was to reproduce the results of the 2019 paper [*Hierarchical attention networks for information extraction from cancer pathology reports*](https://www.sciencedirect.com/science/article/pii/S0933365719303562) on our own dataset of 200k cancer pathology reports pulled from the University of Miami's hospital system.

I set out to train classifiers to predict the diagnosis code ( [ICD-10](https://www.icd10data.com/ICD10CM/Codes/C00-D49)) given the text of a cancer pathology note. This capability serves as proof-of-concept for extracting other information from the reports (stage, grade, tumor size, etc.).

This repository contains:
- *technical report*, which details the scope, work, and findings of the project.
- */presentations/*, which contains two end-of-semester presentations.
- */data/*, which contains sample data that can be used to test models.
- */utilities/*, which contains auxiliary functions for text preprocessing and data management.
- */unsupervised learning/*, which contains clustering and other analysis of pathology notes regarding breast and lung cancer.
- */supervised learning/*, which contains models for both classical and deep learning approaches.

### Hierarchical Attention Networks

A hierarchical attention network is a deep learning model composed of hierarchies of bidirectional LSTMs/GRUs with attention mechanisms. The model has two "hierarchies". The lower hierarchy takes in one sentence at a time, broken into word embeddings. This hierarchy outputs a weighted sentence embedding based on the words in the sentence that are most relevant to the classification. The upper hierarchy takes in one document at a time, broken into the sentence embeddings from the lower hierarchy. This hierarchy outputs a weighted document embedding based on the sentences in the document that are most relevant to the classification. Dropout is applied to this final document embedding, and it is then fed into a softmax classifier.

### Acknowledgements
All funding for this project provided by the University of Miami's Institute for Data Science and Computing (IDSC).

This repository is maintained by Matthew Rossi matthew.rossi@miami.edu. 
