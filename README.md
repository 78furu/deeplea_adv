# Advanced machine learning lab

### Generative Modeling by Estimating Gradients of the Data Distribution

Reproducing [article](https://arxiv.org/abs/1907.05600).
#### Abstract
We introduce a new generative model where samples are produced via Langevin dynamics using gradients of the data distribution estimated with score matching. Because gradients can be ill-defined and hard to estimate when the data resides on low-dimensional manifolds, we perturb the data with different levels of Gaussian noise, and jointly estimate the corresponding scores, i.e., the vector fields of gradients of the perturbed data distribution for all noise levels. For sampling, we propose an annealed Langevin dynamics where we use gradients corresponding to gradually decreasing noise levels as the sampling process gets closer to the data manifold. Our framework allows flexible model architectures, requires no sampling during training or the use of adversarial methods, and provides a learning objective that can be used for principled model comparisons. Our models produce samples comparable to GANs on MNIST, CelebA and CIFAR-10 datasets, achieving a new state-of-the-art inception score of 8.87 on CIFAR-10. Additionally, we demonstrate that our models learn effective representations via image inpainting experiments. 

## Plan (might change):
- [x] Create working envornment 
- [x] Get network
- [x] Modifiy network (see article A.3):
	- [x] Adjustments
    - [x] Run
- [ ] Data preparation:
	- [x] Define noise scales
	- [ ] Instance norm
- [ ] Train network
- [x] Implement Langevin simulated annealing
- [ ] Combine Langevin for diff noise
- [ ]  (Image impainting)

## Reference 
This project is supposed to recreate the methods based in [Song & Ermon's article](https://github.com/ermongroup/ncsn).