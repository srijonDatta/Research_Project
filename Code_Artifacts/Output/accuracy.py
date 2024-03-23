from sklearn.metrics import accuracy_score
import numpy as np

gt_file = np.genfromtxt('causal.gt', delimiter='\t')
gt = gt_file[:, 3:]

cnn_out = np.genfromtxt('cnn.pred', delimiter='\t')
pred_cnn = cnn_out[:, 4:]
accu_cnn = accuracy_score(gt, pred_cnn)
print('Accuracy obtained from CNN model : ', round(accu_cnn, 4))

bert_out = np.genfromtxt('bert.pred', delimiter='\t')
pred_bert = bert_out[:, 4:]
accu_bert = accuracy_score(gt, pred_bert)
print('Accuracy obtained from BERT model : ', round(accu_bert, 4))
