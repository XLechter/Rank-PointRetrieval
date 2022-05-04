#!/bin/bash
#
#python evaluate_model.py --model minkloc --dataset_prefix residential --with_reg --reg 'or' --lamda1 1.0 --lamda2 0.8
#python evaluate_model.py --model minkloc --dataset_prefix university --with_reg --reg 'or' --lamda1 1.0 --lamda2 0.8
#python evaluate_model.py --model minkloc --dataset_prefix business --with_reg --reg 'or' --lamda1 1.0 --lamda2 0.8
#python evaluate_model.py --model minkloc --dataset_prefix oxford --with_reg --reg 'or' --lamda1 1.0 --lamda2 0.2
python evaluate_model.py --model minkloc --dataset_prefix oxford --with_reg --reg 'vc' --lamda1 0.3 --lamda2 0.7
python evaluate_model.py --model lpd --dataset_prefix oxford --with_reg --reg 'vc' --lamda1 0.3 --lamda2 0.7
#python evaluate_model.py --model minkloc --dataset_prefix oxford