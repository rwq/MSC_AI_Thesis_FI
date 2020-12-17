# MSC_AI_Thesis_FI

Repository for my MSc AI thesis @ UvA Amsterdam about Video Frame Interpolation.


In short, the code that was used to find the hyper parameters can be found in  train_hyper_param_search.py.

After this, we used both train_full_run_l1.py and train_full_run_lf.py to create our 

<img src="https://latex.codecogs.com/gif.latex?O_t=$\mathcal{L}\_1$- and $\mathcal{L}\_f$" /> -model respectively.

We omitted the datasets from this repository becaus of the space they require. We did leave the directory structure in place, so these datasets can simply be extracted here.

The Large Motion VI Dataset was created using create_lmd_dataset.py. It takes as input a set of text files which can be found in /created_datasets/large_motion_dataset/.
