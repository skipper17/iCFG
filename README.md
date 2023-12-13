# Inner Classifier-Free Guidance and Its Taylor Expansion for Diffusion Models
This is the implementation of [Inner Classifier-Free Guidance and Its Taylor Expansion for Diffusion Models](https://openreview.net/forum?id=0QAzIMq32X) (ICLR submission paper)

## Overview
Classifier-free guidance (CFG) is a pivotal technique for balancing the diversity and fidelity of samples in conditional diffusion models. This approach involves utilizing a single model to jointly optimize the conditional score predictor and unconditional score predictor, eliminating the need for additional classifiers. It delivers impressive results and can be employed for continuous and discrete condition representations. However, when the condition is continuous, it prompts the question of whether the trade-off can be further enhanced. Our proposed inner classifier-free guidance (ICFG) provides an alternative perspective on the CFG method when the condition has a specific structure, demonstrating that CFG represents a first-order case of ICFG. Additionally, we offer a second-order implementation, highlighting that even without altering the training policy, our second-order approach can introduce new valuable information and achieve an improved balance between fidelity and diversity for Stable Diffusion.

## Main Results
<img width="795" alt="image" src="https://github.com/skipper17/second-order-CFG/assets/36984150/792c767b-237b-438c-960e-6028b8e3839d">

## Second-Order ICFG Sampling

`python sample.py`
