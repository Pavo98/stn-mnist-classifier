import itertools
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def conf_matrix_to_figure(cm: torch.Tensor, num_classes=10, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    assert cm.size() == (num_classes, num_classes)
    if normalize:
        cm = cm.float() / cm.sum(dim=0)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(im, cax=cax)
    ax.set_title(title)
    tick_marks = list(range(num_classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max().item() / 2
    for i, j in itertools.product(range(num_classes), range(num_classes)):
        ax.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.relim()
    fig.tight_layout(pad=2)
    return fig


if __name__ == '__main__':
    conf_matrix = torch.randint(0, 1000, (10, 10))
    conf_matrix_to_figure(conf_matrix).show()
