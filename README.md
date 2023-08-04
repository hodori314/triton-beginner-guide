# Triton Beginner's Guide
In this repository, we will explore Triton, a powerful deep learning optimization library developed by NVIDIA. 
Whether you are new to Triton or eager to enhance your understanding, this guide will walk you through the fundamentals, features, and practical applications of Triton. 
Get ready to embark on an exciting journey to harness the potential of Triton for your machine learning projects.  

Also we utilize the Elastic Net machine learning model using scikit-learn (sklearn). 
Elastic Net is a powerful regression technique that combines both L1 and L2 regularization, offering a balance between Lasso (L1) and Ridge (L2) regression. 
By employing the Elastic Net model, we aim to achieve better generalization and feature selection capabilities for our predictive tasks.

## Train the model
### Initialize Python env to Train Elastic Net
In this project, we need following python packages.
```
numpy
scikit-learn 
skl2onnx
tritonclient[http]
```

We prepared friendly env files for you. You can use `init.sh` to bring up env variable and the above packages.
```
$ git clone https://github.com/hodori314/triton-beginner-guide.git
$ source init.sh
```

### Train Elastic Net
official docs: [(sklearn)elastic-net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)

We train elastic net using scikit-learn packages. However, one thing that you have to notice is that you cannot directly save and use sklearn model for triton server.
Therefore, you must convert scikit-learn packages to torch[sk2torch](https://github.com/unixpickle/sk2torch/) or onnx model[sk2onnx](https://github.com/onnx/sklearn-onnx).
In this project, we use onnx model.

```
$ python train.py
```

To apply for your project, you need to pay attention to the `initial_type` and `final_type` in `train.py`.
You can find 

## Upload traind model into Triton Server

### Construct `model_repository` folder

### Run Triton Server

## Call API Request for Triton Server
In triton, they support multiple language for API calls(i.e., C++, python). In this project, we consider C++ and python.

### Python

### C++

## Appendix
### How to send requests Container(triton server) to Container(triton client)

## Reference