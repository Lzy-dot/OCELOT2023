import torch
import numpy as np


def compute_miou(logits, labels, num_classes):
    # 将logits转换为预测的类别标签
    predictions = torch.argmax(logits, dim=1)
    union = np.zeros([num_classes])
    intersection = np.zeros([num_classes])
    
    for class_label in range(num_classes):
        pred_mask = (predictions == class_label)
        label_mask = (labels == class_label)
        ignore = (labels==255)
        pred_mask = pred_mask & (~ignore)
        label_mask = label_mask & (~ignore)
        intersection[class_label] = (pred_mask & label_mask).sum().float()
        union[class_label] = (pred_mask | label_mask).sum().float() 
        # iou = intersection / (union + 1e-7)  # 加上一个小的常数，避免除以0

        # iou_values.append(iou)

    # 计算mIoU
    # miou = torch.mean(torch.tensor(iou_values))

    return union, intersection




# numpy 版本

#设标签宽W，长H
def fast_hist(a, b, n):#a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测特征图，形状(H×W,)；n是类别数目
    k = (a >= 0) & (a < n) #k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别（去掉了背景）,假设0是背景
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):#分别为每个类别（在这里是19类）计算mIoU，hist的形状(n, n)
    '''
	核心代码
	'''
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))#矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)

def compute_mIoU(pred,label, n_classes):
    hist = np.zeros((n_classes, n_classes))#hist初始化为全零，在这里的hist的形状是[n_classes, n_classes]
    hist += fast_hist(label.flatten(), pred.flatten(), n_classes) #对一张图片计算 n_classes×n_classes 的hist矩阵，并累加
    
    mIoUs = per_class_iu(hist)#计算逐类别mIoU值
    for ind_class in range(n_classes):#逐类别输出一下mIoU值
        print(str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))#在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    return mIoUs