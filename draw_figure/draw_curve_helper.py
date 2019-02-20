# coding:utf-8

'''

this script aims to evalute model and draw p-r curve and roc curve .
Author: chenglong Yi
date: 2019-02-15

'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc 
from scipy import interp
from itertools import cycle


from matplotlib.pyplot import cm



class DrawCurveHelper(object):
    def __init__(self):
        pass
    
    def read_result_file(self, filepath):

        '''
        @param filepath: str
        @rtype: list, list
        '''

        # format : label \t prob \n  single class
    
        y_label = []
        y_score = []

        with open(filepath, "r") as fo:
            lines = fo.readlines()

        cols = len(lines[0].split("\t"))
        
        print("cols:", cols)
        
        for item in lines:
            t_arr = item.split("\t")
            # single class 
            if cols == 2:
                y_label.append(int(t_arr[0]))
                y_score.append(float(t_arr[1]))
        
        return y_label, y_score


    def read_result_file_multi(self, filepath, multicls=False):

        '''
        @param filepath: str
        @param multicls: bool

        @rtype: (list, list)
        '''

        # format : label \t label \t ... \t label # prob \t prob \t ... \t prob \n    multi-class
        y_label = []
        y_score = []

        if multicls == False:
            return


        with open(filepath, "r") as fo:
            lines = fo.readlines()
        
        for item in lines:
            t_arr = item.split("#")
            t_label_arr = t_arr[0].split("\t")
            t_score_arr = t_arr[1].split("\t")

            t_label_list = []
            t_score_list = []

            for item in t_label_arr:
                t_label_list.append(int(item))

            for item in t_score_arr:
                t_score_list.append(float(item))

            
            y_label.append(t_label_list)
            y_score.append(t_score_list)

        
        y_label = np.array(y_label)
        y_score = np.array(y_score)
            
        return y_label, y_score



    def draw_pr_curve(self, filepath):
        '''
        draw single class PR curve.

        @param filepath : str

        '''

        y_label, y_score = self.read_result_file(filepath)
    
        precision, recall, thresholds = precision_recall_curve(
            y_label, y_score)
        

        pr_auc = auc(recall, precision)

        plt.plot(precision, recall, color='darkorange',
                lw=2, label='PR curve (area = %0.2f)' % pr_auc)
        plt.title("Precision/Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend(loc="upper right")
        plt.show()
        plt.savefig("p-r-curve-single-cls.png")


    def draw_pr_curve_multi(self, filepath, multicls=False):
        '''
        draw multi class PR curve.

        @param filepath: str
        @param multicls: bool
        
        '''
        y_label, y_score = self.read_result_file_multi(filepath, True)

        if multicls == False:
            print("invalid parameter.")
            return

        n_classes = len(y_score[0])

        print("n_classes:", n_classes)

        # 计算每一类的PR
        pdic = {}
        rdic = {}
        pr_auc = {}
        for i in range(n_classes):
            pdic[i], rdic[i], _ = precision_recall_curve(y_label[:,i], y_score[:,i])
            pr_auc[i] = auc(rdic[i],pdic[i])
            

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        # auto generate color
        color_auto=iter(cm.rainbow(np.linspace(0,1,n_classes)))
        lw = 2

        for i, color in zip(range(n_classes), colors):
            c = next(color_auto)
            plt.plot(pdic[i], rdic[i], color=c, lw=lw,
                    label='PR curve of class {0} (area = {1:0.2f})'
                    ''.format(i, pr_auc[i]))
        


        plt.title("Precision/Recall Curve-multicls")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend(loc="upper right")
        plt.show()
        plt.savefig("p-r-curve-multi-cls.png")
        

    def draw_roc_curve(self, filepath):
        """
        @param filepath : str

        """

        y_label, y_score = self.read_result_file(filepath)    
            
        # Compute ROC curve and ROC area for each class
        fpr,tpr,threshold = roc_curve(y_label, y_score) ###计算真正率和假正率

        roc_auc = auc(fpr,tpr) ###计算auc的值

        lw = 2
       
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic result')
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig("roc-curve-single-cls.png")


    def draw_roc_curve_multi(self, filepath, multicls=False):
        '''
        @param filepath : str
        @param multicls : bool

        '''

        y_label, y_score = self.read_result_file_multi(filepath, True)

        if multicls == False:
            return

        n_classes = len(y_score[0])

        # print("y_label:", y_label)
        # print("y_label[:,0]:", y_label[:, 0])
        # print("n_classes:", n_classes)
        # print("y_score[:, 0]", y_score[:, 0])

        # 计算每一类的ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_label[:,i], y_score[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area（方法二）
        fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


        
        # Compute macro-average ROC curve and ROC area（方法一）
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        


        # Plot all ROC curves
        lw=2
       
        plt.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        
        color_auto=iter(cm.rainbow(np.linspace(0,1,n_classes)))
        for i, color in zip(range(n_classes), colors):
            c = next(color_auto)
            plt.plot(fpr[i], tpr[i], color=c, lw=lw,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig("ROC curve multi-cls.png")

    









