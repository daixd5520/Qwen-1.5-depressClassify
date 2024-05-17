cm =[[5067  , 11],[ 376 ,  14]]
TP = cm[0][0] #预测为1（正），实际为1（正）
FP = cm[0][1] #预测为1（正），实际为0（负）
FN = cm[1][0] #预测为0（负），实际为1（正）
TN = cm[1][1] #
precision = TP / (TP + FP)
recall = TP / (TP + FN)

print("Precision:", precision)
print("Recall:", recall)