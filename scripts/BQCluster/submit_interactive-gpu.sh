#!/bin/bash
srun -p gpu --gres=gpu:1 --mem=50G --pty /bin/bash