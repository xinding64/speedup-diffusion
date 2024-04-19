## Table of Contents
- [0. Overview](#0-Overview)
- [1. Methodology Improvement](#1-Methodology-Improvement)
  <!-- - [Variational Gap Optimization](#Variational-Gap-Optimization) -->
  <!-- - [Dimension Deduction](#Dimension-Deduction) -->
  - [1.1 Optimized Sampling Efficiency](#11-Speed-up-sample)
    - [1.1.1 Training Scheme](#111-Training-Scheme)
    - [1.1.2 Training-Free Sampling](#112-Training-Free-Sampling)
  - [1.2 Optimized Sampling Efficiency](#12-Speed-up-Structural)
    - [1.2.1 Optimized Structural Efficiency](#121-Mixed-Modeling)
  - [1.3 Optimized timestep Efficiency](#13-Speed-up-timestep)


## 0. Overview


## 1. Methodology Improvement
<!-- ### Variational Gap Optimization -->
<!-- ### Dimension Deduction -->

Nowadays, the main concern of the diffusion model is to speed up its speed and reduce the cost of computing. In general cases, it takes thousands of steps for diffusion models to generate a high-quality sample. Mainly focusing on improving sampling speed, many works from different aspects come into reality. 


### 1.1 Speed-up
#### 1.1.1 Training Scheme

***Knowledge DIstillation***
 - **Progressive distillation for fast sampling of diffusion models**
    - Salimans, Tim and Ho, Jonathan. *ICLR 2022*. [[pdf]](https://arxiv.org/abs/2202.00512) [[code]](https://github.com/google-research/google-research/tree/master/diffusion_distillation) <!-- TODO: Update version -->

 - **ProDiff: Progressive Fast Diffusion Model For High-Quality Text-to-Speech**
    - Huang, Rongjie and Zhao, Zhou and Liu, Huadai and Liu, Jinglin and Cui, Chenye and Ren, Yi. *ACM MM 2022*. [[pdf]](https://arxiv.org/abs/2207.06389) [[code]](https://github.com/Rongjiehuang/ProDiff) <!-- TODO: Update version -->

 - **Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed**
    - Luhman, Eric and Luhman, Troy. *Arxiv 2021* [[pdf]](https://arxiv.org/abs/2101.02388) [[code]](https://github.com/tcl9876/Denoising_Student) <!-- TODO: Cite in text -->


***Diffusion Scheme Learning***
 - **Accelerating Diffusion Models via Early Stop of the Diffusion Process**
    - Lyu, Zhaoyang and Xu, Xudong and Yang, Ceyuan and Lin, Dahua and Dai, Bo. *Arxiv 2022* [[pdf]](https://arxiv.org/abs/2205.12524) [[code]](https://github.com/zhaoyanglyu/early_stopped_ddpm)
 
 - **Truncated diffusion probabilistic models**
    - Zheng, Huangjie and He, Pengcheng and Chen, Weizhu and Zhou, Mingyuan. *Arxiv 2022* [[pdf]](https://arxiv.org/abs/2202.09671)

 - **Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise**
    - Bansal, Arpit and Borgnia, Eitan and Chu, Hong-Min and Li, Jie S. and Kazemi, Hamid and Huang, Furong and Goldblum, Micah and Geiping, Jonas and Goldstein, Tom. *Arxiv 2022* [[pdf]](https://arxiv.org/abs/2208.09392) [[code]](https://github.com/arpitbansal297/cold-diffusion-models)

 - **How Much is Enough? A Study on Diffusion Times in Score-based Generative Models**
    - Franzese, Giulio and Rossi, Simone and Yang, Lixuan and Finamore, Alessandro and Rossi, Dario and Filippone, Maurizio and Michiardi, Pietro. *Arxiv 2022* [[pdf]](https://arxiv.org/abs/2206.05173)

 - **Poisson Flow Generative Models**
   - Xu, Yilun and Liu, Ziming and Tegmark, Max Tegmark and Jaakkola, Tommi. *NeurIPS 2022* [[pdf]](https://arxiv.org/abs/2209.11178) [[code]](https://github.com/Newbeeer/Poisson_flow)

 - **PFGM++: Unlocking the Potential of Physics-Inspired Generative Models**
   - Xu, Yilun Xu and Liu, Ziming and Tian, Yonglong and Tong, Shangyuan Tong and Tegmark, Max Tegmark and Jaakkola, Tommi. *ICML 2023* [[pdf]](https://arxiv.org/abs/2302.04265) [[code]](https://github.com/Newbeeer/pfgmpp)

 - **Stable Target Field for Reduced Variance Score Estimation in Diffusion Models**
   - Xu, Yilun and Tong, Shangyuan and Jaakkola, Tommi. *ICLR 2023* [[pdf]](https://arxiv.org/abs/2302.00670) [[code]](https://github.com/Newbeeer/stf)
 

#### 1.1.2 Training-Free Sampling
***Analytical Method***
 - **Analytic-dpm: an analytic estimate of the optimal reverse variance in diffusion probabilistic models**
    - Bao, Fan and Li, Chongxuan and Zhu, Jun and Zhang, Bo. *Arxiv 2022*. [[pdf]](https://arxiv.org/abs/2201.06503) [[code]](https://github.com/baofff/Analytic-DPM)

***Implicit Sampler***
 - **Denoising Diffusion Implicit Models**
    - Song, Jiaming and Meng, Chenlin and Ermon, Stefano. *ICLR 2020*. [[pdf]](https://arxiv.org/abs/2010.02502) [[code]](https://github.com/ermongroup/ddim)

 - **gDDIM: Generalized denoising diffusion implicit models**
    - Zhang, Qinsheng and Tao, Molei and Chen, Yongxin. *Arxiv 2022*. [[pdf]](https://arxiv.org/abs/2206.05564) [[code]](https://github.com/qsh-zh/gDDIM)
 
 - **Maximum Likelihood Training of Implicit Nonlinear Diffusion Models**
    - Kim, Dongjun and Na, Byeonghu and Kwon, Se Jung and Lee, Dongsoo and Kang, Wanmo and Moon, Il-Chul. *Arxiv 2022*. [[pdf]](https://arxiv.org/abs/2205.13699) 

***Differential Equation Solver Sampler***
- **Fast Sampling of Diffusion Models with Exponential Integrator**
    - Zhang, Qinsheng and Chen, Yongxin. *Arxiv 2022* [[pdf]](https://arxiv.org/abs/2204.13902) [[code]](https://github.com/qsh-zh/deis)

- **Pseudo numerical methods for diffusion models on manifolds**
    - Liu, Luping and Ren, Yi and Lin, Zhijie and Zhao, Zhou. *ICLR 2022*. [[pdf]](https://arxiv.org/abs/2202.09778) [[code]](https://github.com/luping-liu/PNDM) <!-- TODO: Update version -->

- **DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps**
    - Lu, Cheng and Zhou, Yuhao and Bao, Fan and Chen, Jianfei and Li, Chongxuan and Zhu, Jun. *Arxiv 2022* [[pdf]](https://arxiv.org/abs/2206.00927) [[code]](https://github.com/luchengthu/dpm-solver) 

- **Gotta Go Fast When Generating Data with Score-Based Models**
    - Jolicoeur-Martineau, Alexia and Li, Ke and Piché-Taillefer, Rémi and Kachman, Tal and Mitliagkas, Ioannis. *Arxiv 2022* [[pdf]](https://arxiv.org/abs/2105.14080) [[code]](https://github.com/AlexiaJM/score_sde_fast_sampling) 

***Dynamic Programming Adjustment***
- **Learning Fast Samplers for Diffusion Models by Differentiating Through Sample Quality**
    - Watson, Daniel and Chan, William and Ho, Jonathan and Norouzi, Mohammad. *ICLR 2022*. [[pdf]](https://arxiv.org/abs/2202.05830) 

- **Learning to efficiently sample from diffusion probabilistic models**
    - Watson, Daniel and Ho, Jonathan and Norouzi, Mohammad and Chan, William. *Arxiv 2021*. [[pdf]](https://arxiv.org/abs/2106.03802)  

### 1.2 Speed-up-Structural
#### 1.2.1 Optimized Structural Efficiency
*Acceleration Mixture* 
- **Structural pruning for diffusion models**
    - Gongfan Fang, Xinyin Ma, and Xinchao Wang. *In Advances in Neural Information Processing Systems, 2023* [[pdf]](https://arxiv.org/abs/2305.10924) 

- **Snapfusion: Text-to-image diffusion model on mobile devices**
    - Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys,Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren *Arxiv 2023*. [[pdf]](https://arxiv.org/abs/2306.00980) 

- **Diffusion probabilistic model made slim**
    - Xingyi Yang, Daquan Zhou, Jiashi Feng, and Xinchao Wang *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2023*. [[pdf]](https://ieeexplore.ieee.org/xpl/conhome/1000147/all-proceedings)  

- **DiffuseVAE: Efficient, Controllable and High-Fidelity Generation from Low-Dimensiona Latents**
    - Pandey, Kushagra and Mukherjee, Avideep and Rai, Piyush and Kumar, Abhishek. *Arixv 2022*. [[pdf]](https://arxiv.org/pdf/2201.00308) [[code]](https://github.com/kpandey008/DiffuseVAE) 

- **Diffusion normalizing flow**
    - Zhang, Qinsheng and Chen, Yongxin. *NIPS 2021*. [[pdf]](https://arxiv.org/abs/2110.07579) [[code]](https://github.com/qsh-zh/DiffFlow) 



### 1.3 Speed-up-timestep
#### 1.3.1 Optimized Structural Efficiency
*Acceleration Mixture* 
- **Learning to Efficiently Sample from Diffusion Probabilistic Models**
    - Daniel Watson, Jonathan Ho, Mohammad Norouzi, William Chan. *Arixv 2021* [[pdf]](https://arxiv.org/abs/2106.03802) 

- **AutoDiffusion: Training-Free Optimization of Time Steps and Architectures for Automated Diffusion Model Acceleration**
    - Li L, Li H, Zheng X, et al *cvpr 2023*. [[pdf]](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_AutoDiffusion_Training-Free_Optimization_of_Time_Steps_and_Architectures_for_Automated_ICCV_2023_paper.pdf) 

- **AdaDiff: Adaptive Step Selection for Fast Diffusion**
    - Hui Zhang, Zuxuan Wu, Zhen Xing, Jie Shao, Yu-Gang Jiang *Arixv 2024*. [[pdf]](https://arxiv.org/abs/2311.14768)  
