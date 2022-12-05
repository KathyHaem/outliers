%%%%%%%%%% CODE IN 'analysis_code' %%%%%%%%%%

- anisotropy.py: performs anisotropy analysis (as done in chapter 6.2.1)

- cosine_analysis.py: performs parallel data cosine analysis (e.g. for establishing similarity-harming outlier group)

- find_extended_outliers.py: used for establishing expanded set of outliers (as in chapter 6.1.1)

- find_outliers.py: used to 
	a) identify magnitude-wise outliers for an XLM-R layer (note: only layer 8 sentence embeddings are on this CD) 
	b) identify magnitude-wise outliers per each language (used in chapter 5) + generate visualizations (as in 5.1.1)

- plots.py: generates all kinds of visualizations (for details see code itself)

- spearman_and_cosines.py: performs spearman ranking and cosine values analyses (as in chapter 5.3)



%%%%%%%%%% CODE IN 'google_colab' %%%%%%%%%%

All that was performed on Google Colab, i.e. extracting sentence embeddings from XLM-R and
other models, fine-tuning on other tasks and evaluating. Certain code is taken from the XTREME
benchmark repo: https://github.com/google-research/xtreme

- run_tatoeba.ipynb and run_bucc.ipynb: evaluate models on the tasks. Dimensions to remove beforehand can be specified.

- /finetune: code for fine-tuning on XNLI and PAWS-X. Note: fie-tuned model checkpoints (zip files on this CD) need to be moved to MA/finetune/outputs-temp first.