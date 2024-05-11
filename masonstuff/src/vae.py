import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import os


class VAE(nn.Module):
    

    def __init__(self, in_channels=1, latent_dim=128, hidden_dims = [32, 64, 128, 256, 512, 512, 512]):
        super().__init__()

        # Encoder
        self.hidden_dims = hidden_dims

        encoder_modules = []
        for h_dim in hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*encoder_modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # Decoder
        decoder_modules = []
        
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)
        hidden_dims.reverse()
        print(hidden_dims)
        for i in range(len(hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i+1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )
        
        self.decoder = nn.Sequential(*decoder_modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1], 
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=1,
                                      kernel_size=3, padding=1),
                            nn.Tanh())
    
    def encode(self, x):
        x = self.encoder(x)
        print(x.shape)
        x = torch.flatten(x, start_dim=1)
        print(x.shape)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return [mu, log_var]

    def decode(self, x):
        x = self.decoder_input(x)
        x = x.view(-1, self.hidden_dims[0], 2, 2)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x
    
    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.rand_like(std)
        return eps * std + mu
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        return [self.decode(z), mu, log_var]


def loss_fcn(out, labels, kld_weight=0.00025):
    predictions = out[0]
    mu = out[1] 
    log_var = out[2]

    MSE_loss = F.mse_loss(predictions, labels)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    loss = MSE_loss + kld_weight * kld_loss
    return loss, [loss.item(), MSE_loss.item(), kld_loss.item()]



def makedir_if_needed(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



if __name__=="__main__":

    parser = argparse.ArgumentParser()

    # Save locations
    parser.add_argument("--inputs_path", default = "/viscam/projects/audio_nerf/transfer/devocalization/data/processed_framewise/imitations_spec_256.npy")
    parser.add_argument("--labels_path", default = "/viscam/projects/audio_nerf/transfer/devocalization/data/processed_framewise/reference_spec_256.npy")
    parser.add_argument("--base_save_dir", default = "/viscam/projects/audio_nerf/transfer/devocalization/masonstuff/models")
    parser.add_argument("--save_name")

    # Hyperparameters
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()
    batch_size = args.batch_size

    save_dir = os.path.join(args.base_save_dir, args.save_name)
    makedir_if_needed(save_dir)

    # Loading Data, Train/Test Split
    np.random.seed(0)
    input_data = np.load(args.inputs_path)
    label_data = np.load(args.labels_path)
    input_data = torch.from_numpy(input_data).cuda()
    label_data = torch.from_numpy(label_data).cuda()

    print(input_data.dtype, flush=True)
    print(label_data.dtype, flush=True)

    n_total = input_data.shape[0]
    n_train = int(0.8*n_total)
    n_valid = int(0.1*n_total)

    indices = np.arange(n_total)
    np.random.shuffle(indices)

    train_indices = indices[:n_train]
    valid_indices = indices[n_train:n_train+n_valid]
    test_indices = indices[n_train+n_valid:]

    np.save(os.path.join(save_dir, "train_indices.npy"), train_indices)
    np.save(os.path.join(save_dir, "valid_indices.npy"), valid_indices)
    np.save(os.path.join(save_dir, "test_indices.npy"), test_indices)

    print(f"Train, Valid, Test Input Sizes:\n{train_indices.shape}\n{valid_indices.shape}\n{test_indices.shape}", flush=True)


    # Loading Model and Optimizer
    model = VAE().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    # Training Loops
    N_iters_per_epoch = n_train//batch_size

    loss_log = []
    valid_losses = []

    for epoch in range(args.n_epochs):

        print(f"Epoch:\t{epoch}", flush=True)

        train_indices = np.random.shuffle(train_indices)

        for i in range(N_iters_per_epoch):
            batch_indices = train_indices[i*batch_size:(i+1)*batch_size]
            inputs = input_data[batch_indices]
            labels = label_data[batch_indices]

            optimizer.zero_grad()
            out = model(inputs)
            loss, to_log = loss_fcn(out, labels)
            loss.backward()
            optimizer.step()
            loss_log.append(to_log)
            print(loss.item(), flush=True)


        # Computing Validation Loss
        valid_loss_list = []
        weights = []

        with torch.no_grad():

            n_valid_batches = n_valid//batch_size

            for i in range(n_valid_batches):
                valid_batch_indices = valid_indices[i*batch_size:(i+1)*batch_size]
                out = model.forward(input_data[valid_batch_indices])
                predictions = out[0]
                mse_loss = F.mse_loss(predictions, label_data[valid_batch_indices])
                valid_loss_list.append(mse_loss.item())
                weights.append(batch_size)

            if n_valid % batch_size != 0:
                remaining_indices = valid_indices[n_valid_batches * batch_size:]
                out = model.forward(input_data[remaining_indices])
                predictions = out[0]
                mse_loss = F.mse_loss(predictions, label_data[remaining_indices])
                valid_loss_list.append(mse_loss.item())
                weights.append(n_valid % batch_size)

            valid_loss = np.average(valid_loss_list, weights=weights)
            valid_losses.append([len(loss_log), valid_loss]) # len(loss_log) gives the number of iterations
            print(f"Valid Loss:{valid_loss}", flush=True)

        np.save(os.path.join(save_dir, "loss_log.npy"), loss_log)
        np.save(os.path.join(save_dir, "valid_losses.npy"), valid_losses)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_dir,"weights.pt"))


    # # Final Test Loss
    # test_loss_list = []
    # with torch.no_grad():
    #     for test_idx in test_indices:
    #         prediction = model.forward(input_data[test_idx])
    #         mse_loss = F.mse_loss(prediction, label_data[test_idx])
    #         test_loss_list.append(mse_loss.item())

        
    # print("Final Test MSE:")
    # print(np.mean(test_loss_list), flush=True)












        


