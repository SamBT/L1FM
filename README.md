# L1 Foundation Model Development

For now this is just skeleton code for training models on the dark machines anomaly dataset.

## Getting started
You'll first need to set up the mamba package manager (if not already setup), then create the pytorch training environment specified in `environment.yaml`. Follow [these instruction](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) to install mamba on your cluster. If working on the Harvard Cannon cluster via IAIFI, I recommend you set your mamba install directory to point to a location in your folder in the lab storage area: `/n/holystore01/LABS/iaifi_lab/Users/YOUR_USERNAME/`. If you install it in your home area, you will probably run out of disk quota pretty quickly.

Once you've set up mamba, set up the pytorch environment as follows:
```
# first start an interactive job on a GPU node
# this is necessary to properly install the cuda-enabled versions of torch etc.
salloc -p iaifi_gpu -c 1 --time=01:00:00 --mem=16G --gres=gpu:1

# now install the mamba environment
mamba env create -f environment.yaml
```
Once this is done, you should be good to go!

## Training dataset
We'll start by training on the Level 1 anomaly challenge dataset. I've uploaded it to cannon in the directory `/n/holystore01/LABS/iaifi_lab/Lab/sambt/ADChallenge_L1`. There are several different datasets included:
```
Ato4l_lepFilter_13TeV_filtered.h5
background_for_training.h5
BlackBox_background_mix.h5
hChToTauNu_13TeV_PU20_filtered.h5
hToTauTau_13TeV_PU20_filtered.h5
leptoquark_LOWMASS_lepFilter_13TeV_filtered.h5
```
All of these files contain data from simulated collision events from various signal/background processes. Everything not labeled "background" is a BSM signal, and was used as a target for anomaly detection. The background samples are composed of a mixture of Standard Model physics processes (e.g. QCD, top quark, W, Z, etc.). 

Each file is in HDF5 format and the relevant entry is `Particles`, which stores an array of shape `(N,19,4)`. `N` is the number of events, 19 is the number of particles saved per event, and 4 is the number of input features for each particle. The input features are ($p_T$, $\eta$, $\phi$, $c$), where $c$ encodes the particle type (1 = missing energy, 2 = electron, 3 = muon, 4 = jet). The 19 elements are organized as follows (indexing from 0):
1. **Entry 0**:  missing transverse momentum $p_T^\text{miss}. This is not technically a "particle" but it has a $p_T$ and $\phi$ entry (by definition there is no $\eta$, so this is just always set to 0)
2. **Entries 1-4**: 4 electrons
3. **Entries 5-8**: 4 muons
4. **Entries 9-18**: 10 jets

Not every event has 4 electrons, 4 muons, and 10 jets, so any "extra" entries are just filled with zeros (e.g. if there are only 3 electrons in event 0, then `Particles[0,4,:] = [0,0,0,0]`).

## Training setup
This is up to you! I have included an implementation of Particle Transformer (ParT) in `models/ParT.py`, but it will probably need some minor tweaks/adaptation to work with this dataset. 