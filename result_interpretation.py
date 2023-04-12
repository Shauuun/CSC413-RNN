from sklearn.metrics import confusion_matrix
import pandas as pd
data = pd.read_csv('test_result.csv')
data.head()

data['labels'] = data['labels'].apply(lambda x: int(x.split("(")[1].split(")")[0]))
data['pred'] = data['pred'].apply(lambda x: int(x.split('[')[1].split(']')[0]))

# Successful Example
success = data.loc[data['labels'] == data['pred']]
print("Successful Example: ")
print(success.reset_index()['labels'][0])
print(success.reset_index()['inputs'][0])

# Unsuccessful Example
unsuccess = data.loc[data['labels'] != data['pred']]
print("Unsuccessful Example: ")
print(unsuccess.reset_index()['labels'][0])
print(unsuccess.reset_index()['inputs'][0])

# Test Accuracy
print(str(len(success)/len(data)))

# Confusion Matrix
cm = confusion_matrix(data['labels'], data['pred'])
print(cm)

# True positives (TP)
tp = cm[0][0]

# False positives (FP)
fp = cm[0][1]

# False negatives (FN)
fn = cm[1][0]

# True negatives (TN)
tn = cm[1][1]

# Accuracy
accuracy = (tp + tn) / (tp + fp + fn + tn)
print("Accuracy: {:.5f}".format(accuracy))

# FPR
fpr = fp / (fp + tn)
print("False Positive Rate: {:.5f}".format(fpr))

# FNR
fnr = fn / (fn + tp)
print("False Negative Rate: {:.5f}".format(fnr))

# TPR
tpr = tp / (tp + fn)
print("True Positive Rate: {:.5f}".format(tpr))

# TNR
tnr = tn / (tn + fp)
print("True Negative Rate: {:.5f}".format(tnr))