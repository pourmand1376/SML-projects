import sys
import torch
import random
import pathlib
import argparse
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from utils import get_cifar10_model, prepare_cifar10_model2, get_cifar10_data

from visualization import heatmap_grid

base_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, base_path.as_posix())


def plot_attribution(a, ax_, preds, title, cmap='seismic', img_shape=28):
    ax_.imshow(a)
    ax_.axis('off')

    cols = a.shape[1] // (img_shape + 2)
    rows = a.shape[0] // (img_shape + 2)
    for i in range(rows):
        for j in range(cols):
            ax_.text(28 + j * 30, 28 + i * 30, preds[i * cols + j].item(), horizontalalignment="right",
                     verticalalignment="bottom", color="lime")
    ax_.set_title(title)


def main(args):
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    model = get_cifar10_model()
    prepare_cifar10_model2(args.device, model, epochs=args.epochs, train_new=args.train_new)
    model = model.to(args.device)
    model.eval()
    train_loader, test_loader = get_cifar10_data(transform=torchvision.transforms.ToTensor(),
                                                 batch_size=args.batch_size)

    X = []
    Y = []
    for i in range(10):
        j = 0
        for x, y in test_loader:
            if j == 3:
                break
            for z in range(len(x)):
                if j == 3:
                    break
                if y[z] == i:
                    X.append(x[z].numpy())
                    Y.append(y[z].item())
                    j = j + 1

    X = torch.tensor(X).to(args.device)
    Y = torch.tensor(Y).to(args.device)
    X.requires_grad_(True)

    with torch.no_grad():
        y_hat = model(X)
        pred = y_hat.max(1)[1]

    def compute_and_plot_explanation2():

        X.grad = None
        rule = 'epsilon'
        pattern = None
        y_hat = model.forward(X, explain=True, rule=rule, pattern=pattern)
        y_hat = y_hat[torch.arange(X.shape[0]), y_hat.max(1)[1]]
        y_hat = y_hat.sum()

        y_hat.backward()
        attr = X.grad
        attr = heatmap_grid(attr, cmap_name='seismic')

        for i in range(30):
            plt.figure()
            plt.imshow(attr[i])
            plt.title(f'GT = {classes[Y[i].item()]} Pred = {classes[pred[i]]}')
            plt.savefig(f'cifar10_{int(i / 3)}_lrp_explain_{int(i % 3)}.png')

    _std = torch.tensor([0.2023, 0.1994, 0.2010], device=args.device).view((3, 1, 1))
    _mean = torch.tensor([0.4914, 0.4822, 0.4465], device=args.device).view((3, 1, 1))

    for i in range(30):
        plt.figure()
        q = X[i] * _std + _mean
        plt.imshow(np.transpose(q.to('cpu').detach().numpy(), (1, 2, 0)))
        plt.title(f'GT = {classes[Y[i].item()]} Pred = {classes[pred[i]]}')
        plt.savefig(f'cifar10_{int(i / 3)}_lrp_original_{int(i % 3)}.png')

    compute_and_plot_explanation2()

    correct_l1 = 0
    correct_l2 = 0
    total = 0
    ju = 0
    for x, y in test_loader:
        X = torch.tensor(x).to(args.device)
        Y = torch.tensor(y).to(args.device)
        X.requires_grad_(True)

        with torch.no_grad():
            y_hat = model(X)
            pred = y_hat.max(1)[1]

        correct1 = pred == Y
        correct_l1 += sum(correct1.to('cpu').numpy())
        X.grad = None
        rule = 'epsilon'
        pattern = None
        y_hat = model.forward(X, explain=True, rule=rule, pattern=pattern)
        y_hat = y_hat[torch.arange(X.shape[0]), y_hat.max(1)[1]]
        y_hat = y_hat.sum()

        y_hat.backward()
        attr = X.grad
        attr = heatmap_grid(attr, cmap_name='seismic')
        c = np.dot(attr[..., :3], [0.2, 0.3, 0.5])

        n = len(attr)
        X_c = np.zeros((n, 3, 32, 32))

        total += n
        for i in range(n):
            l = np.sort(c[i], axis=None)
            q = l[205]
            e = c[i].copy()
            e[e > q] = 1
            e[e <= q] = 0

            e = 1 - e
            X_c[i] = X[i].to('cpu').detach().numpy() * e
        X_c = torch.tensor(X_c).to(args.device).float()
        with torch.no_grad():
            y_hat_c = model(X_c)
            pred_c = y_hat_c.max(1)[1]

        correct2 = pred_c == Y
        correct_l2 += sum(correct2.to('cpu').numpy())

        q = X_c[0] * _std + _mean
        ju = ju + 1
        plt.imshow(np.transpose(q.to('cpu').detach().numpy(), (1, 2, 0)))
        plt.savefig(f'rrt{ju}.png')

    print(f'accuracy before remove: {correct_l1 / total * 100 :.3f} % ')
    print(f'accuracy after remove: {correct_l2 / total * 100 :.3f} %')

    print(f'total {total}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CIFAR LRP Example")
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--train_new', action='store_true', help='Train new predictive model')
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--seed', '-d', type=int)

    args = parser.parse_args()

    if args.seed is None:
        args.seed = int(random.random() * 1e9)
        print("Setting seed: %i" % args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    main(args)