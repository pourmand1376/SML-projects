import sys
import torch
import random
import pathlib
import argparse
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from utils import get_mnist_model, prepare_mnist_model, get_mnist_data

from visualization import heatmap_grid


base_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, base_path.as_posix())

def main(args): 
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = get_mnist_model()
    prepare_mnist_model(args.device, model, epochs=args.epochs, train_new=args.train_new)
    model = model.to(args.device)
    train_loader, test_loader = get_mnist_data(transform=torchvision.transforms.ToTensor(), batch_size=args.batch_size)

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
        rule = 'alpha1beta0'
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
            plt.title(f'GT = {Y[i].item()} Pred = {pred[i]}')
            plt.savefig(f'mnist_{int(i/3)}_lrp_explain_{int(i%3)}.png')



    for i in range(30):
        plt.figure()
        plt.imshow(X[i].to('cpu').detach().numpy().squeeze(), cmap='gray')
        plt.title(f'GT = {Y[i].item()} Pred = {pred[i]}')
        plt.savefig(f'mnist_{int(i / 3)}_lrp_original_{int(i%3)}.png')

    compute_and_plot_explanation2()

    correct_l1 = 0
    correct_l2 = 0
    total = 0

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
        X_c = np.zeros((n, 1, 28, 28))

        total += n
        for i in range(n):
            l = np.sort(c[i], axis=None)
            q = l[156]
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

    print(f'accuracy before remove: {correct_l1/total * 100 :.3f} % ')
    print(f'accuracy after remove: {correct_l2/total * 100 :.3f} %')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("MNIST LRP Example")
    parser.add_argument('--batch_size', type=int, default=3000)
    parser.add_argument('--train_new', action='store_true', help='Train new predictive model')
    parser.add_argument('--epochs', '-e', type=int, default=5)
    parser.add_argument('--seed', '-d', type=int)

    args = parser.parse_args()

    if args.seed is None: 
        args.seed = int(random.random() * 1e9)
        print("Setting seed: %i" % args.seed)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    main(args)
