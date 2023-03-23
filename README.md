# LDMIC

The official PyTorch implementation of our **ICLR 2023** paper: 

**LDMIC: Learning-based Distributed Multi-view Image Coding**

[Xinjie Zhang](https://xinjie-q.github.io/), [Jiawei Shao](https://shaojiawei07.github.io/), [Jun Zhang](https://eejzhang.people.ust.hk/)

[[ICLR Open Review](https://openreview.net/forum?id=ILQVw4cA5F9)] [[ArXiv Preprint](https://arxiv.org/abs/2301.09799)]]

### :bookmark:Brief Introduction

Multi-view image compression plays a critical role in 3D-related applications. Existing methods adopt a predictive coding architecture, which requires joint encoding to compress the corresponding disparity as well as residual information. This demands collaboration among cameras and enforces the epipolar geometric constraint between different views, which makes it challenging to deploy these methods in distributed camera systems with randomly overlapping fields of view. Meanwhile, distributed source coding theory indicates that efficient data compression of correlated sources can be achieved by independent encoding and joint decoding, which motivates us to design a **learning-based distributed multi-view image coding** (LDMIC) framework. With independent encoders, LDMIC introduces a simple yet effective joint context transfer module based on the cross-attention mechanism at the decoder to effectively capture the global inter-view correlations, which is insensitive to the geometric relationships between images. Experimental results show that LDMIC significantly outperforms both traditional and learning-based MIC methods while enjoying fast encoding speed. 

## Acknowledgement

:heart::heart::heart:Our idea is implemented based on the following projects. We really appreciate their wonderful open-source works!

- [CompressAI](https://github.com/InterDigitalInc/CompressAI) [[related paper](https://arxiv.org/abs/2011.03029)]
- [Checkerboard Context Model for Efficient Learned Image Compression](https://github.com/JiangWeibeta/Checkerboard-Context-Model-for-Efficient-Learned-Image-Compression/tree/main) [[related paper](https://arxiv.org/abs/2103.15306v2)]

## Citation

If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.

```
@inproceedings{zhang2023ldmic,
  title={LDMIC: Learning-based Distributed Multi-view Image Coding},
  author={Zhang, Xinjie and Shao, Jiawei and Zhang, Jun},
  booktitle={International Conference on Learning Representations},
  year={2023},
}
```

