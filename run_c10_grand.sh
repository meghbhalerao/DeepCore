for frac in 0.001 0.005 0.01 0.05 0.1 0.2 0.3
do
CUDA_VISIBLE_DEVICES=2 python -u main.py --fraction $frac --dataset CIFAR10 --data_path ~/datasets --num_exp 5 --workers 10 --optimizer SGD -se 10 --selection GraNd --model ResNet9 --lr 0.1 -sp ./result --batch 128
done
