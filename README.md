# Deep Observationfilter

[![Github Repository](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/fholzm/Obs-TasNet)

This contains the checkpoints and code for data generation, training, and evaluation of Obs-TasNet, a deep learning-based observation filter for virtual sensing in active noise control (ANC).

## Description

Local active noise control (ANC) with adaptive processing requires an accurate residual error signal at the point of cancellation, which is often obtained via virtual sensing. We propose Obs-TasNet, a neural approach that estimates observation filter coefficients for the remote microphone technique (RMT) [1] in time-variant settings. By estimating coefficients during operation, the method eliminates the need for pre-optimizing filters for selected scenarios and subsequent interpolation.

Obs-TasNet builds on a modified [inter-channel Conv-TasNet](https://github.com/donghoney0416/IC_Conv-TasNet) [2]. The raw waveform signals from remote microphones, as well as the coordinates of the virtual microphone, are embedded as latent representation using a learnable encoding. A temporal convolutional network (TCN), employing dilated depthwise separable convolutions and an output transformation, predicts the observation filter coefficients. In simulation, the proposed ANC system with RMT and Obs-TasNet achieves superior noise reduction over a substantially wider frequency range than a multi-point ANC baseline, validating the effectiveness of observation filter estimation.

## Structure

The repository is structured as follows:

- `train_and_evaluate.py`: Code for training the Obs-TasNet model and evaluating its performance on the validation set.
- `test_and_export.py`: Code for testing the evaluating model on the test set and exporting the relevant metrics.
- `start_training.sh`: Shell script for starting and detaching the training process with the path to the config file as the first argument.
- `start_test.py`: Script for rendering the test set for relevant configurations from the article, calls `test_and_export.py`.
- `utils/`: Utility classes and modules in Python
- `data_generation/`: Code for generating the datasets used for training, validation, and testing.
- `evaluation/`: Code for evaluating the performance of the trained model on the test set.
- `configs/`: Configuration files for training, evaluation, and data generation.
- `checkpoints/`: Directory with model checkpoints and weights.
- `export/`: Directory with results as `.npz` and `.pkl` files for evaluation in Python.
- `data/`, `figures/`, `logs/`, `tensorboard/`: Directories for storing data, figures, logs, and tensorboard files.

## Usage

1. Clone the repository as well as the checkpoints via git LFS or download the whole reporitory on Zenodo.
2. Install the necessary python packages, e.g., via `pip install -r requirements.txt`.
3. Generate the dataset (rougly 1 TB) by navigating into the `data_generation/` directory and running `main_datagen.py`. Acoustic scenes are simulated in [TASCAR](https://tascar.org/). For ease of use, a [development container](https://containers.dev/) for use in VS Code with all necessary dependencies is provided. The generated dataset is stored in the `data/` directory.
4. Evaluate the model by running `start_test.py` to render the test set for relevant configurations from the article. This calls `test_and_export.py` which loads the trained model, performs inference on the test set, and exports the relevant metrics as `.npz` and `.pkl` files in the `export/` directory, as well as the raw audio files in `rendered_testset/`.
5. Evaluate and plot the results by running the scripts in the `evaluation/` directory in the denoted order.

## Citation

The article associated with this repository is currently under review. Citation information will be updated once the article is published.

## License

This project is licensed under the [MIT License](LICENSE).

## References

[1] A. Roure and A. Albarrazin, “The remote microphone technique for active noise control,” in INTER-NOISE 1999, (Fort Lauderdale FL, USA), pp. 1233–1244, Dec. 1999.

[2] D. Lee, S. Kim, and J.-W. Choi, “Inter-channel Conv-TasNet for multichannel speech enhancement,” Nov. 08, 2021, arXiv: arXiv:2111.04312. doi: 10.48550/arXiv.2111.04312.
