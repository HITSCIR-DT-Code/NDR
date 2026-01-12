# ReadMe

The code for NDR.

1.Put the dataset file in to **data/**. For the dataset, please refer to

> Ganguli D, Lovitt L, Kernion J, et al. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned[J]. arXiv preprint arXiv:2209.07858, 2022.

2.Put your model files in the **models/**, like:

> models/T5-base

3.Run

```shell
conda create --name xxx python=3.9
pip install -r requirements.txt
```



4.Run generate.py for training and inference



Here's a simple illustration for code files. For more information, please see the annotations:

- model.py 
  - the whole model structure 
- generate.py
  - run it to the implement the training and inference phase model
  - for inference use, adjust the parameter L to control the decoder input ration. 


