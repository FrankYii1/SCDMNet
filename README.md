# SCDMNet
It is based on U2PL.It is a semi-supervised contrastive learning remote sensing semantic segmentation network with dual-attention and multi-level feature fusion.
# Usage
# Train
cd experiments/pascal/1464/ours
# use torch.distributed.launch
sh train.sh <num_gpu> <port>
# After training, the model should be evaluated by
sh eval.sh
