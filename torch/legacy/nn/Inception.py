import math
import torch
from torch.legacy.nn import ReLU, SpatialMaxPooling, DepthConcat, Sequential, SpatialConvolution
from .Module import Module


class Inception(Module):
    def __init__(self, config):
        self.inputSize = config.inputSize
        self.outputSize = config.outputSize
        self.reduceSize = config.reduceSize
        self.reduceStride = config.reduceStride or (1,1)
        self.transfer = config.transfer or ReLU()
        self.batchNorm = config.batchNorm
        self.padding = True
        if config.padding:
            self.padding = config.padding

        self.kernelSize = config.kernelSize or (5,3)
        self.kernelStride = config.kernelStride or (1,1)
        self.poolSize = config.poolSize or 3
        self.poolStride = config.poolStride or 1
        self.pool = config.pool or SpatialMaxPooling(self.poolSize, self.poolSize, self.poolStride, self.poolStride)

        iWidth, iHeight = 100, 200
        oWidth, oHeight = 0, 0

        depthConcat = DepthConcat(2)
        for i in range(len(self.kernelSize)):
            mlp = Sequential()
            reduce_ = SpatialConvolution(self.inputSize,
                                         self.reduceSize[i], 1, 1,
                                         self.reduceStride[i] or 1,
                                         self.reduceStride[i] or 1)
            mlp.add(self.transfer.clone())

            pad = self.padding and math.floor(self.kernelSize[i]/2) or 0

            conv = SpatialConvolution(
                self.reduceSize[i], self.outputSize[i],
                self.kernelSize[i], self.kernelSize[i],
                self.kernelStride[i], self.kernelStride[i])
            mlp.add(conv)
