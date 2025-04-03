import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
import h5py
import numpy as np

from models.networks import MLP, DeepSetsEncoder, SimCLRModel
from losses import SupervisedSimCLRLoss
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from plotting import make_corner

# a really basic example script for setting up the dataloader and training a contrastive embedding

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device {device}')

    # load in the datasets
    # just doing this in memory; will break for larger datasets where you need to load on-the-fly
    data_dir = "/n/holystore01/LABS/iaifi_lab/Lab/sambt/ADChallenge_L1/"
    file_bkg = "background_for_training.h5"
    files_sig = ["Ato4l_lepFilter_13TeV_filtered.h5",
                 "hChToTauNu_13TeV_PU20_filtered.h5",
                 "hToTauTau_13TeV_PU20_filtered.h5",
                 "leptoquark_LOWMASS_lepFilter_13TeV_filtered.h5"]
    
    ## loading data: just take <=20k events from each file
    N_load = 20_000
    # load in the backgrounds
    with h5py.File(data_dir + file_bkg, 'r') as f:
        bkg_data = f['Particles'][:N_load] # shape (N, 19, 4), 19 particles w/ 4 features per event
        bkg_mask = np.all(bkg_data==0,axis=-1) # all zeros as features means no particle, should be masked
        bkg_labels = np.zeros(bkg_data.shape[0])
    print(f"Loaded backgrounds with {len(bkg_data)} events")
    
    # combine the signals
    sig_data = []
    sig_masks = []
    sig_labels = []
    for i,file in enumerate(files_sig):
        with h5py.File(data_dir + file, 'r') as f:
            sig = f['Particles'][:N_load]
            mask = np.all(sig==0,axis=-1)
            labels = (i+1)*np.ones(sig.shape[0])
            sig_data.append(sig)
            sig_masks.append(mask)
            sig_labels.append(labels)
        print(f"Loaded signals {file} with {len(sig)} events")
    sig_data = np.concatenate(sig_data,axis=0)
    sig_masks = np.concatenate(sig_masks,axis=0)
    sig_labels = np.concatenate(sig_labels,axis=0)
    
    # combine it all
    data = np.concatenate([bkg_data,sig_data],axis=0)
    masks = np.concatenate([bkg_mask,sig_masks],axis=0)
    labels = np.concatenate([bkg_labels,sig_labels],axis=0)
    perm = np.random.permutation(data.shape[0])
    data = data[perm]
    masks = masks[perm]
    labels = labels[perm]
    print("Total events: ", len(data))

    # normalize
    for i in range(data.shape[1]):
        # normalize each particle independently 
        data[:,i] = (data[:,i] - np.mean(data[:,i],axis=0)) / np.std(data[:,i],axis=0)

    # split datasets with 80/10/10 split
    data_train, data_val, labels_train, labels_val, masks_train, masks_val = train_test_split(data, labels, masks, train_size=0.8)
    data_val, data_test, labels_val, labels_test, masks_val, masks_test = train_test_split(data_val, labels_val, masks_val, train_size=0.5)

    # tensorize
    data_train = torch.tensor(data_train).float()
    data_val = torch.tensor(data_val).float()
    data_test = torch.tensor(data_test).float()
    masks_train = torch.tensor(masks_train).float()
    masks_val = torch.tensor(masks_val).float()
    masks_test = torch.tensor(masks_test).float()
    labels_train = torch.tensor(labels_train).float()
    labels_val = torch.tensor(labels_val).float()
    labels_test = torch.tensor(labels_test).float()

    # make torch datasets
    train = TensorDataset(data_train, masks_train, labels_train)
    val = TensorDataset(data_val, masks_val, labels_val)
    test = TensorDataset(data_test, masks_test, labels_test)

    # make dataloaders
    train_loader = DataLoader(train, batch_size=512, shuffle=True)
    val_loader = DataLoader(val, batch_size=512, shuffle=True)
    test_loader = DataLoader(test, batch_size=512, shuffle=False)

    # instantiate deep sets model
    dim_space = 4
    simclr_temp = 0.1
    phi_net = MLP(input_dim=4,
                  hidden_dims=[64, 64],
                  output_dim=64)
    F_net = MLP(input_dim=64,
                  hidden_dims=[128, 128],
                  output_dim=dim_space)
    projection_net = MLP(input_dim=dim_space,
                  hidden_dims=[dim_space],
                  output_dim=dim_space)
    encoder = DeepSetsEncoder(phi=phi_net, f=F_net)
    model = SimCLRModel(encoder=encoder, projector=projection_net).to(device)
    
    # set up training
    lr = 1e-3
    num_epochs = 50
    loss_fn = SupervisedSimCLRLoss(temperature=simclr_temp, base_temperature=simclr_temp).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    pbar = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    best_loss = 99999.0
    best_state = None
    for i in pbar:
        model.train()
        epoch_losses = []
        for batch in train_loader:
            data, masks, labels = batch
            data = data.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            features = F.normalize(model(data, mask=masks),dim=1).unsqueeze(1) # normalize for simclr loss, shape (B,1,D) needed 
            loss = loss_fn(features, labels=labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)
        
        model.eval()
        epoch_losses = []
        with torch.no_grad():
            for batch in val_loader:
                data, masks, labels = batch
                data = data.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                features = F.normalize(model(data, mask=masks),dim=-1).unsqueeze(1) # normalize for simclr loss
                loss = loss_fn(features, labels=labels)
                epoch_losses.append(loss.item())
        val_loss = np.mean(epoch_losses)
        val_losses.append(val_loss)
        
        pbar.set_postfix_str(f"Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{i}.pt")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()

    # save the best model
    torch.save(best_state, "checkpoints/best_model.pt")
    model.load_state_dict(best_state)

    # make a plot
    plt.figure(figsize=(8,6))
    plt.plot(1+np.arange(len(train_losses)), train_losses, label='train')
    plt.plot(1+np.arange(len(val_losses)), val_losses, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("checkpoints/training.png")
    plt.close()

    # evaluate on test set
    print("Evaluating on test set...")
    model.eval()
    embeddings = []
    embedding_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            data, masks, labels = batch
            data = data.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            features = model.embed(data, mask=masks).cpu().numpy()
            embeddings.append(features)
            embedding_labels.append(labels.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    embedding_labels = np.concatenate(embedding_labels, axis=0)

    # save the embeddings
    with h5py.File("embeddings.h5", 'w') as f:
        f.create_dataset('embeddings', data=embeddings)
        f.create_dataset('labels', data=embedding_labels)

    # make a corner plot
    make_corner(embeddings, embedding_labels, save=True)
    
    print("all done!")

if __name__ == '__main__':
    main()