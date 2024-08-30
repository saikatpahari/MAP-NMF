import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


class Metrics():
    y = []
    y_pre = []

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def getFscAccNmiAri(self):
        y_true = self.y_true
        y_pred = self.y_pred
        Fscore, Accuracy = self.getFscoreAndAcc()
        NMI = normalized_mutual_info_score(y_true, y_pred)
        ARI = adjusted_rand_score(y_true, y_pred)
        return Fscore, Accuracy, NMI, ARI

    def getFscoreAndAcc(self):
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        n = len(y_pred) 
        assert len(y_pred) == len(y_true), "len(pred) not equal to len(y_true)"
        true = np.unique(y_true)
        pred = np.unique(y_pred)
        true_size = len(true)  #
        pred_size = len(pred)  

       
        a = np.ones((true_size, 1), dtype=int) * y_true  #
        b = true.reshape(true_size, 1) * np.ones((1, n), dtype=int)
        pid = (a == b) * 1  # true_size by n
        a = np.ones((pred_size, 1), dtype=int) * y_pred
        b = pred.reshape(pred_size, 1) * np.ones((1, n), dtype=int)
        cid = (a == b) * 1  # pred_size by n
        confusion_matrix = np.matmul(pid, cid.T)  
        #        P    N
        #   P   TP   FN
        #   N   FP   TN
      
        temp = np.max(confusion_matrix, axis=0)  
        Accuracy = np.sum(temp, axis=0) / float(n)

        # f-score
        ci = np.sum(confusion_matrix, axis=0)  # [TP+FP,FN+TN]  
        pj = np.sum(confusion_matrix, axis=1)  # [TP+FN,FP+TN]  
        precision = confusion_matrix / (np.ones((true_size, 1), dtype=float) * ci.reshape(1, len(ci)))
        
        recall = confusion_matrix / (pj.reshape(len(pj), 1) * np.ones((1, pred_size), dtype=float))

        F = 2 * precision * recall / (precision + recall)
        F = np.nan_to_num(F)
        temp = (pj / float(pj.sum())) * np.max(F, axis=0)
        Fscore = np.sum(temp, axis=0)
        return Fscore, Accuracy
