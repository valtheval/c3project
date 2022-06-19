# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Model developped by Valentin M.
- Developped on May 15th 2022
- 2nd version of this model
- Model type is : Random Forest implemented in sklearn
- Trained on census data using train test splitting strategy (20% test), no hyperparameter tunning

## Intended Use
- Inference on salary based on census data. The model wants to predict the salary of a population such as there are two classes >50K and <=50k

## Training Data
- 80% of the dataset (random split), 26048 rows, 108 columns

## Evaluation Data
- 20% of the dataset (random split), 6513 rows, 108 columns

## Metrics
- Model assessed using Precision, Recall and f1 score

## Ethical Considerations
- No sensitive data, the data is anonymized

## Caveats and Recommendations
- Should be used carefully since it's the 2nd version of model without hyperparamter tunning. This is an exercise on a benchmark dataset, you can use it to learn basics of data science and MLOps.
