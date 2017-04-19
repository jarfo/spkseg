#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python train.py --data_dir data/IV1 --vector_size 400 --savefile IV1_nospk
# Epoch 1/1. Validation loss: 1.14611220093. Accuracy: 0.951086283351
# Perplexity on test set:  1.14553265006 Accuracy:  0.951321566603
CUDA_VISIBLE_DEVICES=1 python evaluate.py --model cv/IV1_nospk --vocabulary data/IV1/vocab.npz  --init init.npy --itext data/IV1/test.ctm  --otext test_IV1_nospk.prb --calc --beam 1
./evaluate.sh test_IV1_nospk.prb
# Conversations: 40
# No speaker change (0): 63142
# Speaker change    (1): 4648
# Precision (ppv): 74.18 %
# Recall    (tpr): 44.51 %
# Miss rate (fnr): 55.48 %
# Fall-out  (fpr): 1.14 %
# Accuracy       : 95.13 %
