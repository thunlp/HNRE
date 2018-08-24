import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

model = 'cnn'

label = './data/all_true_label.npy'
score = sys.argv[1]
y_true = np.load(label)
y_score = np.load(score)


auc = average_precision_score(y_true, y_scores)
print('AUC: {0:1.4f}'.format(auc))

precision,recall,threshold = precision_recall_curve(y_true,y_scores)

plt.plot(recall[:], precision[:], lw=2, label=model)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.3, 1.0])
plt.xlim([0.0, 0.4])
plt.title('Precision-Recall Area={0:1.4f}'.format(auc))
plt.legend(loc="upper right")
plt.grid(True)
plt.show()
