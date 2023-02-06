This is the repository for our paper "Exploring anisotropy and outliers in multilingual language models for cross-lingual semantic sentence similarity".
You can use the code to reproduce our results, or adapt it for further experiments.

# Installation

Using a new virtual environment, you can install libraries from the requirements file.
Check what versions of torch/CUDA and faiss you need, and run:

```bash
pip install -r src/requirements.txt
```

# Usage

A large set of experiments is bundled together in ``main.py``.
Specifically, this includes for Tatoeba, STS, and Wiki data: 
Extracting sentence embeddings, outlier and anisotropy analysis, and plotting the mean embeddings.
For Tatoeba and STS, task scores are calculated as well.
tSNE visualisations are commented out from ``main.py`` just because they take a long time.

You can either call ``main.py`` directly, or run experiments individually.
BUCC2018 is not included in the ``main.py``; you can run this task on different models and with zeroed-out dimensions using ``run_bucc2018.py``. 
Check the command line help for how to run it.

Similarly, extracting sentence embeddings, outlier and anisotropy analysis, and plots can all be run individually using the respective command line interfaces.

## For Figure 1



# License

We release this repository under the MIT license, see LICENSE.md.
We state where we have adapted code from other repositories in the relevant places.


# Citation

None yet.
