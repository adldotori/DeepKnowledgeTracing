# DeepKnowledgeTracing
source code for the paper Deep Knowledge Tracing. http://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf

# Usage

usage: python main.py [-m (`train`, `infer`, `seq_op`) : mode] [-e : epoch] [-b : batch_size]

# Environment
Hardware  
* macOS Catalina v10.15.7

Framework  
* torch 1.7.0
* torchvision 0.8.1
* scikit-learn 0.23.2
* numpy 1.16.5
* pandas 1.1.2
* tqdm 4.54.1

# Result
## RNN
hyperparameter  
* batch_size : `100`
* epoch : `20`

time : 1hour  
ROC_AUC_SCORE : **0.84347**

## LSTM
