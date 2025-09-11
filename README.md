# CrayOnSegmentation

This repo is forked from [NanoMask](https://github.com/bowang-lab/NanoMASK) and used in combination of our in-house tumor segmentation model. Huge thanks to Bowang Lab for making the model publicly available! The NanoMask component of this package remains unchanged from the original release.

# Installation

```bash
mkdir -p /platforms/radiomics/
cd /platforms/radiomics/
git clone https://github.com/LiYouBioinfo/CRayOnSegmentation.git
cd CRayOnSegmentation
```

```bash
# Prepare virtual env
python3.9 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install   torch==1.12.1+cu116   torchvision==0.13.1+cu116   torchaudio==0.12.1   --index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116

# Install binary greedy 1.3.0-alpha directly into .venv/bin
curl -L -o /tmp/greedy.tar.gz \
  "https://master.dl.sourceforge.net/project/greedy-reg/Experimental/greedy-1.3.0-alpha-Linux-x86_64.tar.gz?viasf=1" && \
tar -xzf /tmp/greedy.tar.gz -C /tmp && \
mv /tmp/greedy-1.3.0-alpha-Linux-x86_64/bin/* .venv/bin/ && \
rm -rf /tmp/greedy-1.3.0-alpha-Linux-x86_64 /tmp/greedy.tar.gz && \
greedy -h | head -n 3
```

# Download Pretrained Models
```bash
# Download the NanoMask Pretrained Model and the in-house tumor segmentation model from the private AWS S3 repo. Credentials can be found from CrayonAI AWS us-east-1 S3
cd NanoMask/nnunet
aws s3 cp s3://radiomics/nnUNet_data.tar.gz ./
tar xvf nnUNet_data.tar.gz
```

## Acknowledgement

The regular organ segmentation model is based on the [NanoMask](https://github.com/bowang-lab/NanoMASK). Thanks for the Bowang Lab for making the model publicly available!

This NanoMask model is based on the [nnUNet](https://github.com/MIC-DKFZ/nnUNet) framework. Thanks for the nnUNet team very much!
