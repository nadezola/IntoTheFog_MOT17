# Fog Rendering on Private Data (Airport Surveillance)

This branch enables fog augmentation on airport surveillance images.
It supports both homogeneous and heterogeneous fog with varying intensity levels.
For more information, please refer to our paper:

> **[Into the Fog: Evaluating Robustness of Multiple Object Tracking](https://bmvc2024.org/proceedings/362/)**\
Nadezda Kirillova, Muhammad Jehanzeb Mirza, Horst Bischof, Horst Possegger

## Setup

1. Clone the repository (branhc: private_data)
```bash
git clone -b private_data https://github.com/nadezola/IntoTheFog_MOT17.git
```

2. We use:
* Python 3.9
* CUDA 12.1
* pytorch 2.4


3. Working directory is the root of the repository.

4. We leverage the monocular depth estimation pipeline provided 
by the [Hugging Face Transformers library](https://huggingface.co/docs/transformers).

## Run

1. Configure the file `opt.py`.
2. Run:
    ```bash
    python run_augmentation.py --input       # Specify the path to a data split
                               --out         # Specify the path where to save the outputs
                               --loaddepth   # [optional] Load depth images instead of run depth estimation
    ```

##  Citation 
If you find our code useful, please reference:

    @inproceedings{Kirillova_2024_BMVC,
      title = {{Into the Fog: Evaluating Robustness of Multiple Object Tracking}},
      author = {Nadezda Kirillova and Muhammad Jehanzeb Mirza and Horst Bischof and Horst Possegger},
      booktitle = {35th British Machine Vision Conference 2024, {BMVC} 2024, Glasgow, UK, November 25-28, 2024},
      publisher = {BMVA},
      year = {2024},
      url = {https://papers.bmvc2024.org/0362.pdf},
    }