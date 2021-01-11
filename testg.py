import cv2
from operator import xor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
from src.models.model_image_translation import ResUnetGenerator, ResidualBlock, ResidualBlock
import torch_pruning as tp
import matplotlib.pyplot as plt
# model = ResUnetGenerator(input_nc=6, output_nc=3,
#                          num_downs=6, use_dropout=False).to("cuda")
# params = sum([np.prod(p.size()) for p in model.parameters()])
# print("Number of Parameters: %.1fM" % (params/1e6))

# img = cv2.imread('examples_cartoon/wilk.png')
# print(img.shape)

bg = cv2.imread('examples_cartoon/wilk_bg.jpg')
bg = cv2.resize(bg, (839, 919))
cv2.imwrite('examples_cartoon/wilk_bg.jpg', bg)


# print(model)
# img = cv2.imread(
#     r'/home/hack/Downloads/Annotation (1)/Annotation/ROW_1278_11.jpg')
# plt.imshow(img)
# plt.show()
# summary(model, input_size=(6, 256, 256))

# i = 0
# for m in model.modules():

#     if isinstance(m, ResidualBlock):
#         # print((m.block.modules()))
#         for j in m.block.modules():
#             if isinstance(j, nn.Conv2d):
#             i += 1
#                 pass
# print(i)


# def prune_model(model):
#     model.cpu()
#     DG = tp.DependencyGraph().build_dependency(model, torch.randn(5, 6, 256, 256))

#     def prune_conv(conv, pruned_prob):
#         weight = conv.weight.detach().cpu().numpy()
#         out_channels = weight.shape[0]
#         L1_norm = np.sum(np.abs(weight), axis=(1, 2, 3))
#         num_pruned = int(out_channels * pruned_prob)
#         # remove filters with small L1-Norm
#         prune_index = np.argsort(L1_norm)[:num_pruned].tolist()
#         plan = DG.get_pruning_plan(conv, tp.prune_conv, prune_index)
#         plan.exec()

#     block_prune_probs = []
#     for i in range(100):
#         if i < 30:
#             block_prune_probs.append(0.1)
#         if i > 30 and i < 50:
#             block_prune_probs.append(0.2)
#         if i > 50:
#             block_prune_probs.append(0.3)
#     blk_id = 0
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d):
#             prune_conv(m, block_prune_probs[blk_id])
#             blk_id += 1
#         # if isinstance(m, ResidualBlock):
#         #     for j in m.block.modules():
#         #         if isinstance(j, nn.Conv2d):
#         #             prune_conv(j, block_prune_probs[blk_id])
#         #     blk_id += 1

#     return model


# model = prune_model(model).to('cuda')
# params = sum([np.prod(p.size()) for p in model.parameters()])
# print("Number of Parameters: %.1fM" % (params/1e6))
# torch.save(model, 'model.h5')


# x = torch.load('model.h5')

# img = torch.zeros((3, 6, 256, 256)).to('cuda')
# print(x(img))
# summary(x, input_size=(3, 6, 256, 256))
