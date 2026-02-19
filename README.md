
<div align="center">
  <a href="https://github.com/HPCI-Lab">
    <img src="./assets/HPCI-Lab.png" alt="HPCI Lab Logo" width="100" height="100">
  </a>

  <h3 align="center">yProv4ML</h3>

  <p align="center">
    A unified interface for logging and tracking provenance information in machine learning experiments, both on distributed as well as large scale experiments.
    <br />
    <a href="https://hpci-lab.github.io/yProv4ML/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/HPCI-Lab/yProv4ML/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/HPCI-Lab/yProv4ML/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<br />

<div align="center">
  
[![Contributors](https://img.shields.io/github/contributors/HPCI-Lab/yProv4ML?style=for-the-badge)](https://github.com/HPCI-Lab/yProv4ML/graphs/contributors)
[![Forks](https://img.shields.io/github/forks/HPCI-Lab/yProv4ML?style=for-the-badge)](https://github.com/HPCI-Lab/yProv4ML/network/members)
[![Stars](https://img.shields.io/github/stars/HPCI-Lab/yProv4ML?style=for-the-badge)](https://github.com/HPCI-Lab/yProv4ML/stargazers)
[![Issues](https://img.shields.io/github/issues/HPCI-Lab/yProv4ML?style=for-the-badge)](https://github.com/HPCI-Lab/yProv4ML/issues)
[![GPLv3 License](https://img.shields.io/badge/LICENCE-GPL3.0-green?style=for-the-badge)](https://opensource.org/licenses/)

</div>

This library is part of the yProv suite, and provides a unified interface for logging and tracking provenance information in machine learning experiments, both on distributed as well as large scale experiments. 

It allows users to create provenance graphs from the logged information, and save all metrics and parameters to json format.

## Data Model

![Data Model](./assets/prov4ml.datamodel.png)

## Example

![Example](./assets/example.png)

The image shown above has been generated from the [example](./examples/prov4ml_torch.py) program provided in the ```example``` directory.

## Metrics Visualization

![Loss and GPU Usage](./assets/System_Metrics.png)

![Emission Rate](assets/Emission_Rate.png) 

## Experiments and Runs

An experiment is a collection of runs. Each run is a single execution of a machine learning model. 
By changing the ```experiment_name``` parameter in the ```start_run``` function, the user can create a new experiment. 
All artifacts and metrics logged during the execution of the experiment will be saved in the directory specified by the experiment ID. 

Several runs can be executed in the same experiment. All runs will be saved in the same directory (according to the specific experiment name and ID).

# Documentation

For detailed information, please refer to the [Documentation](https://hpci-lab.github.io/yProv4ML/)

# Contributors

- [Gabriele Padovani](https://github.com/lelepado01)
- [Luca Davi](https://github.com/lucadavii)
- [Sandro Luigi Fiore](https://github.com/sandrofioretn)
