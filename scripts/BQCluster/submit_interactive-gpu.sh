#!/bin/bash
srun -p gpu -n 20 --gres=gpu:1 --mem=50G --pty /bin/bash