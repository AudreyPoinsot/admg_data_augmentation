# A Guide for Practical Use of ADMG Causal Data Augmentation

## Requirements

* Python 3.9+
* Pip
* Conda (optional, but recommended)

* See [requirements.txt](./requirements.txt) for the others.

## Installing the dependencies

### Using pip

* Install PyGraphviz following the procedures described into [pygraphviz.github.io/documentation/stable/install.html](https://pygraphviz.github.io/documentation/stable/install.html)
* Execute the following instruction

```bash
pip install -r requirements.txt
```

### Using Conda

* Execute the following instruction

```bash
conda env create -f environment.yml
conda activate causalda_iclr23
```

## Reproducing the experiments

To reproduce the experiments of the paper, you have to:

* Install the required dependencies following the instructions described into the previous section
* run the notebook [Test_causalda_on_simulated_data.ipynb](./experiments/iclr23/Test_causalda_on_simulated_data.ipynb) setting the global variable *use_light_param* to *False* to launch the tests (WARNING: it can take some time)
* run the notebook [Results_test_causalda_on_simulated_data.ipynb](./experiments/iclr23/Results_test_causalda_on_simulated_data.ipynb) setting the global variable *use_paper_results* to *False* to plot your new results

The results of our own experiments are saved in the folder [paper_results](./experiments/iclr23/paper_results) and can be visualized using the notebook [Results_test_causalda_on_simulated_data.ipynb](./experiments/iclr23/Results_test_causalda_on_simulated_data.ipynb) setting the global variable *use_paper_results* to *True*

## Reference
The code in the folder [causal_data_augmentation](./causal_data_augmentation) has been adapted from https://github.com/takeshi-teshima/incorporating-causal-graphical-prior-knowledge-into-predictive-modeling-via-simple-data-augmentation/tree/main/causal_data_augmentation 
The code in the scripts [acyclic_graph_generator.py](./experiments/suite/data/simulations/acyclic_graph_generator.py) and [causal_mechanisms.py](./experiments/suite/data/simulations/causal_mechanisms.py) have been adapted from https://github.com/FenTechSolutions/CausalDiscoveryToolbox/tree/master/cdt/data

## License

Copyright (c) 2023 the original author or authors.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the MIT License (MIT).

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
