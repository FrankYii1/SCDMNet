# SCDMNet
It is based on U2PL.It is a semi-supervised contrastive learning remote sensing semantic segmentation network with dual-attention and multi-level feature fusion.
# Installation
git clone https://github.com/Haochen-Wang409/U2PL.git && cd U2PL
conda create -n u2pl python=3.6.9
conda activate u2pl
pip install -r requirements.txt
pip install pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
# Train
cd experiments/pascal/1464/ours
# use torch.distributed.launch
sh train.sh <num_gpu> <port>
# After training, the model should be evaluated by
sh eval.sh
