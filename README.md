# Generative Modeling for Atmospheric Particle Events

![Python](https://img.shields.io/badge/Python-3.7+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Generative%20Models-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Evaluation-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

## Overview

This project investigates conditional generation of atmospheric particle-event data collected by the **MAGIC Gamma Telescope**. It compares a conditional Wasserstein generative adversarial network (WGAN) with a compact diffusion-based approach for producing synthetic telescope observations conditioned on particle type.

Instead of generating raw telescope images, the models learn a ten-dimensional representation based on **Hillas parameters**—physically meaningful measurements describing the geometry, orientation, concentration, and position of Cherenkov-light clusters.

The complete analysis, saved outputs, visualizations, and model implementations are available in [`MAGIC_Generative_Models.ipynb`](MAGIC_Generative_Models.ipynb).

## Motivation

Gamma-ray photons and hadrons interact with Earth's atmosphere and produce cascades of secondary particles. These cascades emit Cherenkov radiation, which can be observed by ground-based telescopes. Photon- and hadron-initiated events have different geometric signatures, allowing machine-learning models to distinguish between them.

Collecting real observations and running detailed physical simulations can be expensive. A generative model capable of producing realistic class-conditional event vectors could support:

- rapid experimentation with scientific machine-learning pipelines;
- augmentation of limited event classes;
- classifier development and robustness testing;
- exploration of alternatives to computationally intensive simulation.

## Dataset

The project uses the [MAGIC Gamma Telescope dataset](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope) from the UCI Machine Learning Repository.

- **Observations:** 19,020
- **Input features:** 10 continuous Hillas parameters
- **Target classes:** photon (`g`) and hadron (`h`)
- **Project encoding:** photon = `1`, hadron = `0`

| Feature | Description |
| --- | --- |
| Length | Major axis of the event ellipse |
| Width | Minor axis of the event ellipse |
| Size | Logarithm of total pixel intensity |
| Conc | Concentration in the two brightest pixels |
| Conc1 | Concentration in the brightest pixel |
| Asym | Projected distance from the brightest pixel to the center |
| M3Long | Third-moment descriptor along the major axis |
| M3Trans | Third-moment descriptor along the minor axis |
| Alpha | Orientation angle of the major axis |
| Dist | Distance from the event center to the camera center |

## Project workflow

1. Load and inspect the MAGIC event data.
2. Create a stratified train–test split.
3. Normalize long-tailed features with `QuantileTransformer`.
4. Train a class-conditional WGAN.
5. Train a compact diffusion-based neural network.
6. Return generated samples to the original feature space.
7. Compare real and synthetic marginal distributions.
8. Evaluate realism with independent logistic-regression and gradient-boosting classifiers.
9. Quantify real-versus-generated separability using accuracy and ROC AUC.

## Models

### Conditional WGAN

The generator receives a latent noise vector together with the particle-class label and produces ten synthetic event features. The discriminator receives an event vector and its label. Training uses multiple discriminator updates per generator update and weight clipping following the original WGAN strategy.

### Diffusion-based generator

The second experiment applies scheduler-controlled noise to normalized event vectors and trains a class-conditioned multilayer perceptron using mean squared error. This provides a compact tabular-data comparison with adversarial generation.

Both networks use fully connected layers, ReLU activations, and batch normalization.

## Evaluation strategy

Visual overlap between feature histograms is useful but does not capture relationships across the complete ten-dimensional distribution. The notebook therefore uses a **classifier-based two-sample test**:

1. label real observations as `1` and generated observations as `0`;
2. train an independent classifier to separate the two groups;
3. measure its performance on held-out data.

For a balanced dataset, classifier accuracy and ROC AUC near **0.5** are ideal because they indicate that generated and real samples are difficult to distinguish. Scores approaching **1.0** reveal systematic differences.

## Recorded results

| Experiment | Real-vs-generated ROC AUC | Interpretation |
| --- | ---: | --- |
| Conditional WGAN evaluation | 0.718 | The model captures meaningful structure, but synthetic events remain distinguishable from real observations. |
| Second recorded generative evaluation | 0.713 | Similar separability is observed in the second experiment. |

During training, the independent classifier scores generally moved toward chance level, suggesting that the generated samples became progressively more realistic. The final ROC AUC values nevertheless show that both experiments leave detectable differences in the multivariate feature distribution.

## Repository structure

```text
.
├── README.md
├── MAGIC_Generative_Models.ipynb
└── img/
    ├── cgan.png
    ├── clf.png
    ├── gamma_p.png
    ├── geo.jpg
    ├── magic1.jpg
    └── shower.jpg
```

The `img/` directory is required for the scientific diagrams referenced by the notebook. The dataset can be downloaded directly from UCI or attached as a Kaggle dataset.

## Installation

Create a Python environment and install the required packages:

```bash
pip install jupyter numpy pandas matplotlib scikit-learn torch diffusers
```

Clone the repository and start Jupyter:

```bash
git clone <your-repository-url>
cd <repository-name>
jupyter notebook MAGIC_Generative_Models.ipynb
```

The saved notebook was executed in a Kaggle environment and currently references this dataset location:

```text
/kaggle/input/pzad-hw1/2023/hw/hw1/data/magic04.data
```

When running locally, download `magic04.data` from UCI and update the `pd.read_csv(...)` path to its local location.

## Key technologies

- Python
- NumPy and pandas
- Matplotlib
- scikit-learn
- PyTorch
- Hugging Face Diffusers
- Conditional generative modeling
- Classifier-based synthetic-data evaluation

## Limitations and future work

- Replace weight clipping with a gradient-penalty objective for more stable WGAN training.
- Use a linear critic output for a canonical Wasserstein formulation.
- Implement and validate a complete iterative reverse-diffusion sampling process.
- Add multivariate diagnostics such as correlation-matrix comparison and maximum mean discrepancy.
- Tune model depth, latent dimension, learning rates, and training duration.
- Compare downstream particle-classification performance with and without synthetic augmentation.

## References

- [MAGIC Gamma Telescope dataset — UCI](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)
- [MAGIC Collaboration](https://magic.mpp.mpg.de/)
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

## License

Add the license that best matches your intended use before publishing the repository. The dataset is distributed separately by its original provider.
