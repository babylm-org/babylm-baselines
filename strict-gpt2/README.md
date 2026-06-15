BabyLM 2026: GPT-2 Baselines for the Strict, Strict-Small and Multilingual Tracks
=================================================================================

This is the codebase that was used to run training for the GPT-2 baselines of the Strict, Strict-Small and Multilingual tracks of the BabyLM 2026 Challenge. Please refer to the [Call for Papers](https://arxiv.org/pdf/2602.20092) for a description of challenge rules for all relevant tracks.

Installation
------------

To set up the environment, set up a virtual environment with uv (or any other environment manager of your choice) and install the requirements:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install requirements.txt
```

Data Processing
---------------

The repository contains helper scripts to download the raw datasets for each challenge track, process and clean them and then train tokenizers for each one of them. These are meant to be examples and you may change them, improve on them or approach each component completely differently. To replicate the data processing for baseline training, run the following:

```
bash experiment_scripts/download_datasets.sh                                  
bash experiment_scripts/split_all_parquet_datasets.sh			      
bash experiment_scripts/clean_all_datasets.sh				      
bash experiment_scripts/build_all_multilingual.sh			      
     							  		      
bash experiment_scripts/train_all_tokenizers.sh				     
```

In sequence, this
- Downloads the Strict, Strict-Small and nld/zho datasets from BabyLM-community
- Splits the nld/zho dataset .parquet files to data source-specific parquet files to match with the Strict/Strict-Small dataset formats
- Cleans each individual dataset
- Constructs multilingual datasets with different mixes of languages while taking into account each language's byte-premiums
- Trains BPE tokenizers on each constructed dataset

Training
--------
To replicate baseline training, run the following two scripts:

```
bash experiment_scripts/train_monolingual.sh
bash experiment_scripts/train_multilingual.sh	
```

Uploading to HF
---------------
We also have helper scripts for uploading trained models (and their intermediate checkpoints) to Huggingface. The models trained on this repo on the [BabyLM 2026 Baselines](https://huggingface.co/collections/BabyLM-community/babylm-2026-baselines) collection were uploaded with the following:

```
bash experiment_scripts/run_all_uploads.sh
```

As with all of the files in this repository, you can change this to fit your use-case.