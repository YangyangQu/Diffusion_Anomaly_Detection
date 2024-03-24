import argparse
import torch
import functools
from torch.optim import Adam
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from Sde import marginal_prob_std, diffusion_coeff
from Network import ScoreNet
from loss import loss_fn

def main(args):
    device = args.device
    sigma = args.sigma
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr
    checkpoint_name = f'model_n{args.n_epochs}_lr{args.lr}_sigma{args.sigma}.pth'

    # Define the marginal_prob_std and diffusion_coeff functions with the specified sigma
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    # Create the score model and move it to the specified device
    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn)).to(device)

    # Load the MNIST dataset and select specific classes
    dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
    selected_classes = [1,2,3,4,5,6,7,8,9]
    selected_indices = [i for i in range(len(dataset)) if dataset.targets[i] in selected_classes]
    selected_dataset = Subset(dataset, selected_indices)
    data_loader = DataLoader(selected_dataset, batch_size=batch_size, shuffle=True)

    # Define the optimizer
    optimizer = Adam(score_model.parameters(), lr=lr)

    # Training loop
    tqdm_epoch = tqdm(range(n_epochs))
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader:
            x = x.to(device)
            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        # Print the averaged training loss so far.
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

    # Save the trained model checkpoint
    torch.save(score_model.state_dict(), checkpoint_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with hyperparameters')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device for training (cuda or cpu)')
    parser.add_argument('--sigma', type=float, default=25.0,
                        help='Sigma hyperparameter')
    parser.add_argument('--n_epochs', type=int, default=1200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--checkpoint_name', type=str, default='ckptUnet1207.pth',
                        help='Name of the checkpoint file to save')

    args = parser.parse_args()
    main(args)
