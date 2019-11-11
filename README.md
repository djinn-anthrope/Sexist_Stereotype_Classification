# Sexist Stereotype Classification

Due to the advent of social media, there is an increase in the number of interactions on the internet which may be classified as unhealthy or sexist. One of these platforms is Instagram. We use neural models to determine whether a comment or caption on Instagram is sexist or about sexism, and if so, what category it lies in.

| Model                                | Recall | Precision | F1 \-Score | Accuracy |
|--------------------------------------|--------|-----------|------------|----------|
| Single Layer LSTM \(50 dimensions\)  | 0\.59  | 0\.43     | 0\.49      | 0\.70    |
| Single Layer LSTM \(100 dimensions\) | 0\.50  | 0\.39     | 0\.44      | 0\.62    |
| Two Layer LSTM \(100 dimensions\)    | 0\.45  | 0\.36     | 0\.41      | 0\.58    |
| Single Layer RNN \(100 dimensions\)  | 0\.43  | 0\.37     | 0\.39      | 0\.57    |

| Model                                | Recall | Precision | F1 \-Score | Accuracy |
|--------------------------------------|--------|-----------|------------|----------|
| Single Layer LSTM \(50 dimensions\)  | 0\.55  | 0\.49     | 0\.518     | 0\.605   |
| Single Layer LSTM \(100 dimensions\) | 0\.51  | 0\.44     | 0\.472     | 0\.535   |
| Two Layer LSTM \(100 dimensions\)    | 0\.484 | 0\.443    | 0\.462     | 0\.513   |


![](results/sns_classfication/train_loss_allsns.png)

Recall            |  Precision
:-------------------------:|:-------------------------:
![](results/sns_classfication/rec_all_sns.png) | ![](results/sns_classfication/prec_all_sns.png)
Accuracy            |  F1-score
![](results/sns_classfication/acc_all_sns.png) | ![](results/sns_classfication/f1-score-sns.png)


![](results/multiple/mutiple_train_loss_all.png)

Recall            |  Precision
:-------------------------:|:-------------------------:
![](results/multiple/multi_rec_all_sns.png) | ![](results/multiple/multi_prec_all_sns.png)
Accuracy            |  F1-score
![](results/multiple/mutlti_acc_all_sns.png) | ![](results/multiple/mutiple_f1_all.png)





