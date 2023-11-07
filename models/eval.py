import os
import numpy as np
import pandas as pd
import seaborn as sns
import PIL.Image as Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from matplotlib import pyplot
import warnings
warnings.filterwarnings("ignore")

#-----------------------------------------------------------
def show_confusion_matrix(confusion_matrix, class_names):
  cm = confusion_matrix.copy()
  cell_counts = cm.flatten()
  cm_row_norm = cm / cm.sum(axis=1)[:, np.newaxis]
  row_percentages = ["{0:.2f}".format(value) for value in cm_row_norm.flatten()]
  cell_labels = [f"{cnt}\n{per}" for cnt, per in zip(cell_counts, row_percentages)]
  cell_labels = np.asarray(cell_labels).reshape(cm.shape[0], cm.shape[1])
  df_cm = pd.DataFrame(cm_row_norm, index=class_names, columns=class_names)
  hmap = sns.heatmap(df_cm, annot=cell_labels, fmt="", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True Sign')
  plt.xlabel('Predicted Sign')
  plt.show()
#-----------------------------------------------------------

#class_names = ['0','1','2','3','4']
class_names = ['0','1']
data = pd.read_csv("result/test.csv", sep=",") 
y_test = data['correct']
y_pred = data['predict']
print(classification_report(y_test, y_pred, target_names=class_names))
cm = confusion_matrix(y_test, y_pred)
show_confusion_matrix(cm, class_names)
print()

