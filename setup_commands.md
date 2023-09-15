# Pipenv
```bash
sudo apt install pipenv
```

Everything else can probably be installed at a user level in a virtual environment, though it might be nice to have some of these packages globally for others to use

# PyTorch
Since numpy is a prerequisite, it can be installed with
```bash
sudo apt install python-numpy
```
Numpy can optionally be installed with pip directly of course:
```bash
sudo -H pip3 install numpy
```

The torch libraries don't exist as apt packages, but can be installed with the following command:
If using CUDA 11.7 (which I *think* should work on volta, 
```bash
sudo -H pip3 install torch torchvision torchaudio
```
For CPU install (which would not be ideal as volta's GPUs couldn't be used,
```bash
sudo -H pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

# scikit-learn
scikit-learn exists both as an apt package and in pip directly, so it can be installed one of two ways:
```bash
sudo -H pip3 install scikit-learn
```
or
```bash
sudo apt install python3-sklearn
```
