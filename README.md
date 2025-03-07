# L1 Foundation Model Development

For now this is just skeleton code for training models on the dark machines anomaly dataset.

## Getting started
You'll first need to set up the mamba package manager (if not already setup), then create the pytorch training environment specified in `environment.yaml`. Follow [these instruction](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) to install mamba on your cluster. If working on the Harvard Cannon cluster via IAIFI, I recommend you set your mamba install directory to point to a location in your folder in the lab storage area: `/n/holystore01/LABS/iaifi_lab/Users/YOUR_USERNAME/`. If you install it in your home area, you will probably run out of disk quota pretty quickly.

Once you've set up mamba, set up the pytorch environment as follows:
```salloc -p iaifi_gpu -c 1 --time=01:00:00 --mem=16G --gres=gpu:1 # start an interactive job on a GPU node: this is necessary to properly install the cuda-enabled versions of torch etc.
mamba env create -f environment.yaml
```
Once this is done, you should be good to go!