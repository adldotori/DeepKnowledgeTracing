# DeepKnowledgeTracing
source code for the paper Deep Knowledge Tracing. http://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf

## Usage

usage: python main.py [-m (`train`, `infer`, `seq_op`) : mode] [-e : epoch] [-b : batch_size]

## Environment
Hardware  
* macOS Catalina v10.15.7

Framework  
* torch 1.7.0
* torchvision 0.8.1
* scikit-learn 0.23.2
* numpy 1.16.5
* pandas 1.1.2
* tqdm 4.54.1

## Result
### RNN
hyperparameter  
* batch_size : `100`
* epoch : `20`

time : 1hour  
ROC_AUC_SCORE : **0.84347**

### LSTM
hyperparameter  
* batch_size : `100`
* epoch : `20`

time : 6hour
ROC_AUC_SCORE : **0.84674**

### Recommend Sequence

mode : seq_op

Example) After solving 5 problems, recommend more 5 problems

prob ans  
1 O  
1 O  
2 X  
2 X  
2 X  
=>   
77  
69  
77  
69  
69  

### Knowledge Tracing Papers Review
4 models : **BKT, DKT, DKTMN, SAKT**

<http://ai-hub.kr/post/133/>