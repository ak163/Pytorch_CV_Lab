# Plant Pathology 2020 FGVC7

## Introduction

A deep learning model pipeline for training, experimentaiton and deployment for the Kaggle Competition, [Plant Pathology 2020](https://www.kaggle.com/c/plant-pathology-2020-fgvc7/overview), utilising:

- [PyTorch](https://pytorch.org/): A Deep Learning Framework for high-performance AI research
- [Weights and Biases](https://wandb.ai/site): tool for experiment tracking, dataset versioning, and model management
- [Apex](https://nvidia.github.io/apex/): A Library to Accelerate Deep Learning Training using AMP, Fused Optimizer, and Multi-GPU 
- [TensorRT](https://developer.nvidia.com/tensorrt): high-performance neural network inference optimizer and runtime engine for production deployment
- [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server): inference serving software that simplifies the deployment of AI models at scale
- [Streamlit](https://streamlit.io/): framework to quickly build highly interactive web applications for machine learning models

For a quick tutorial about all these modules, check out [tutorials folder](tutorials/). Exploratory data analysis for the same can also be found in the [notebooks folder](notebooks/).

## Structure

```bash
├── app                 # Interactive Streamlit app scripts
├── data                # Datasets
├── examples            # assignment on pytorch amp and ddp
├── model               # Directory to save models for triton
├── notebooks           # EDA, Training, Model conversion, Inferencing and other utility notebooks
├── tutorials           # Tutorials on the modules used
└── requirements.txt    # Basic requirements
```

## Usage

## EDA: Data Evaluation
Data can be explored with various visualization techniques provided in `eda.ipyb` [notebooks folder](notebooks/)

### Training the model

To run the pytorch resnet50 model use `pytorch_train.ipynb`. 

The code is inspired by [Pytorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

Once the model is trained, you can even run model explainabilty using the [shap library](https://shap.readthedocs.io/en/latest/index.html). The tutorial notebook for the same can be found in the [notebooks folder](notebooks/).

### Model Conversion and Inferencing

Once you've trained the model, you will need to convert it to different formats in order to have a faster inference time as well as easily deploy them. You can convert the model to ONNX, TensorRT FP32 and TensorRT FP16 formats which are optimised to run faster inference. You will also need to convert the PyTorch model to TorchScript. Procedure for converting and benchmarking all the different formats of the model can be found in [notebooks folder](notebooks/).

### Model Deployment and Benchmarking

Now your models are ready to be deployed. For deployment, we utilise the Triton Inference Server. It provides an inferencing solution for deep learning models to be easily deployed and integrated with various functionalities. It supports HTTP and gRPC protocol that allows clients to request for inferencing, utilising any model of choice being managed by the server. The process of deployment can be found in [Triton Inference Server.md](tutorials/).

Once your inferencing server is up and running, the next step it to understand as well as optimise the model performance. For this purpose, you can utilise tools like `perf_analyzer` which helps you measure changes in performance as you experiment with different parameters. 

### Interactive Web App

To run the Streamlit app:

```bash
cd app/
streamlit app.py
```

This will create a local server on which you can view the web application. This app contains the client side for the Triton Inference Server, along with an easy to use GUI. 

## Acknowledgement

This repository is built with references and code snippets from the [NN Template](https://github.com/lucmos/nn-template) by [Luca Moschella](https://github.com/lucmos). 
