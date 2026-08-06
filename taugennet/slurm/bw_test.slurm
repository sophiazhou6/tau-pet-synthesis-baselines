#!/bin/bash
#SBATCH --job-name=bw_test --partition=gpu --gres=gpu:nvidia_rtx_pro_6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:05:00
#SBATCH --output=results/logs/bw_test_%j.out --error=results/logs/bw_test_%j.err
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:$PYTHONPATH"
/usr/bin/python3.9 -c "
import torch
print('device:', torch.cuda.get_device_name(0))
print('capability:', torch.cuda.get_device_capability(0))
print('arch_list:', torch.cuda.get_arch_list())
x = torch.randn(256,256,device='cuda'); print('matmul OK, sum=', float((x@x).sum()))
"
