# **Advanced Data Balancing to Improve the Detection of Robot Performance Anomalies** - Suplementary Materials

This repository contains source code and results for experiments related to the article _"Advanced Data Balancing to Improve the Detection of Robot Performance Anomalies"_.

Folder structure:
 - `reports` - tex tables with complete results.
 - `modules/clo` - git submodule for _Class Local Optimization_ algorithm.
 - `experiment.py` - python script for running experiment.

To obtain `clo` algorithms please initialize git submodule:

```
git submodule init
git submodule update --remote
```

Run `experiment.py` script to create `results` directory with pickled dictionaries storing results. Run `results.py` to generate _csv_ file prepared for further processing.
