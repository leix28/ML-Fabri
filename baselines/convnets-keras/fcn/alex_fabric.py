#!/usr/bin/env python
# encoding: utf-8
# File Name: alex_fabric.py
from convnets import AlexNet

def main():
    model = AlexNet('../../../../storage/alexnet_weights.h5', row=64, col=64)
    pass

if __name__ == '__main__':
    main()
