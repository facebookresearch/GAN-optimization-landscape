# Code for [A Closer Look at the Optimization Landscapes of Generative Adversarial Networks](https://arxiv.org/abs/1906.04848)

This is the code for reproducing the experimental results in our paper [A Closer Look at the Optimization Landscapes of Generative Adversarial Networks](https://arxiv.org/abs/1906.04848), Hugo Berard, Gauthier Gidel,  Amjad Almahairi, Pascal Vincent, Simon Lacoste-Julien, 2019.

If you find this code useful please cite our paper:
```
@misc{berard2019closer,
    title={A Closer Look at the Optimization Landscapes of Generative Adversarial Networks},
    author={Hugo Berard and Gauthier Gidel and Amjad Almahairi and Pascal Vincent and Simon Lacoste-Julien},
    year={2019},
    eprint={1906.04848},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

For any questions regarding the code please contact Hugo Berard (berard.hugo@gmail.com).

## License

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

## Running the Code

We provide a conda environment to run the code:
`conda create -f mnist-exp_environment.yml`

The code for computing the eigenvalues and the path-angle is in `plot_path_tools.py`.

To run the code for the Mixture of Gaussian experiment:
`python train_mixture_gan.py OUTPUT_PATH --deterministic --saving-stats`

To run the code for the MNIST experiment:
`python train_mnist.py`

The visualization of the results can be done with `mnist_plots.ipynb`
