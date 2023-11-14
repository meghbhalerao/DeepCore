#!/bin/bash

# Define your four functions
function loop1() {
    for frac in 0.01 0.05
        do
        for sel in GraNd Glister
        do
        CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction $frac --dataset CIFAR100 --data_path ~/datasets --num_exp 5 --workers 10 --optimizer SGD -se 10 --selection $sel --model ResNet9 --lr 0.1 -sp ./result --batch 128
        done
        done
}

function loop2() {
    for frac in  0.1 0.2 0.3
        do
        for sel in GraNd Glister
        do
        CUDA_VISIBLE_DEVICES=1 python -u main.py --fraction $frac --dataset CIFAR100 --data_path ~/datasets --num_exp 5 --workers 10 --optimizer SGD -se 10 --selection Forgetting --model ResNet9 --lr 0.1 -sp ./result --batch 128
        done
        done
}

function loop3() {
    for frac in 0.4 0.5 0.6
        do
        for sel in GraNd Glister
        do
        CUDA_VISIBLE_DEVICES=2 python -u main.py --fraction $frac --dataset CIFAR100 --data_path ~/datasets --num_exp 5 --workers 10 --optimizer SGD -se 10 --selection Forgetting --model ResNet9 --lr 0.1 -sp ./result --batch 128
        done
        done
}

function loop4() {
    for frac in  0.7 0.8 0.9
        do
        for sel in GraNd Glister
        do
        CUDA_VISIBLE_DEVICES=3 python -u main.py --fraction $frac --dataset CIFAR100 --data_path ~/datasets --num_exp 5 --workers 10 --optimizer SGD -se 10 --selection Forgetting --model ResNet9 --lr 0.1 -sp ./result --batch 128
        done
        done
}

# Run each loop in the background
loop1 &
loop2 &
loop3 &
loop4 &

# Wait for all loops to finish
wait
