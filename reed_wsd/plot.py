import matplotlib.pyplot as plt
from reed_wsd.analytics import Analytics


def plot_roc(predictions):
    analytics = Analytics(predictions)
    fpr, tpr, auc = analytics.roc_curve()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUROC = %0.2f' % auc)
    plt.legend(loc='lower right')
    axes = plt.gca()
    axes.set_ylim([-0.05, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def plot_pr(predictions):
    analytics = Analytics(predictions)
    precision, recall, auc = analytics.pr_curve()
    plt.title('Precision-Recall')
    plt.plot(recall, precision, 'b', label = 'AUPR = %0.2f' % auc)
    plt.legend(loc='lower right')
    axes = plt.gca()
    axes.set_ylim([-0.05, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()


def plot_curves(*pycs):
    for i in range(len(pycs)):
        curve = pycs[i][0]
        label = pycs[i][1]
        label = label + "; aupy = {:.3f}".format(curve.aupy())
        curve.plot(label)
    plt.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()
