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

## Upload traind model into Triton Server

### Construct `model_repository` folder

This is directory structure for `model_repository`.
```
# in model_repository 
elastic_example 
├── 1 
│ └── model.onnx 
└── config.pbtxt
```

`model.onnx` is a model file for inference. (You can make this file using `python train.py`.)  
`config.pbtxt` contains meta-data for input and output tensor and its infernce model. You can construct your own `config.pbtxt` by following guide links.  

```
name: "{model name specified in model_repository/}"
backend: "{backend for inference - e.g., pytorch, onnxruntime, tensorflow}"
max_batch_size: {maxinum batch size that model can support}

input [
    {
        name: "{input data name}",
        data_type: {ref to below link},
        dims: [{ref to below link}]
    }
]
output [
    {
        name: "{input data name}",
        data_type: {ref to below link},
        dims: [{ref to below link}]
    }
]
```

[data_type](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#datatypes)
[dims](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#shape-tensors)

### Run Triton Server

Move to the parent folder of the model_repository directory.
```
cd go/to/your/model_repository
cd ..
```  

Create triton container
```
$ docker run --gpus=1 -it --name triton_server --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:22.02-py3
$ tritonserver --model-repository=/models
```

If you want to use multiple gpu, you can change `--gpus` option. - e.g., `--gpus=all`  
If you want to use network for inference in other containers, you can add `--network` option. - e.g., `--network=triton_network` (About using docker network see Appendix)

After typing above commands, you can find your model is ready.
```
...
I0802 01:25:50.743333 57 server.cc:592] 
+-----------------+---------+--------+
| Model           | Version | Status |
+-----------------+---------+--------+
| elastic_example | 1       | READY  |
+-----------------+---------+--------+
...
I0802 01:25:50.889596 57 grpc_server.cc:4375] Started GRPCInferenceService at 0.0.0.0:8001
I0802 01:25:50.890008 57 http_server.cc:3075] Started HTTPService at 0.0.0.0:8000
I0802 01:25:50.932264 57 http_server.cc:178] Started Metrics Service at 0.0.0.0:8002

```

## Call API Request for Triton Server
In triton, they support multiple language for API calls(i.e., C++, python). In this project, we consider C++ and python.

### Python
First, install `tritonclient[http]` in your python/conda environment.  
Second, fix triton server url(line no.6) that is fit to your serevr setting.
If you didnot use `--network` option, then you can use default url for inference.  
Finally, `python client.py` will show you inference result from triton API.  


### C++
First, build c++ library.
```
$ apt-get install zlib1g-dev default-jdk maven
$ git clone https://github.com/triton-inference-server/client.git
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX=`pwd`/path/to/install/c++/library -DTRITON_ENABLE_CC_HTTP=ON ..
$ make cc-clients
```

Second, fix `#include http_client.h` to fit your installed triton c++ library path.  
Third, make binary files for `client.cpp`. You can use `CMakeLists.txt` or `Makefile`.
```
# build using CMakeLists.txt
mkdir build
cd build
cmake ..
cmake --build .
```

```
# build using Makefile
make
```
For executing the binary file, `./myprogram`.


## Appendix
### How to send requests Container(triton server) to Container(triton client)
First, create docker network.
```
docker create network triton_network
```

Second, add network to containers.
If container is already created, then
```
docker network connect triton_network {already created container name}
```

Or you can directly add `--network` option while creating container.
```
docker run ... --network triton_network
```

After you connect network to containers, you can configure the connections using 
```
docker inspect triton_network 
```

Third, use IP address of container for triton inference.
For example, after your `inspect triton_network`, you can find allocated IPv4ADdresses for each container.
```
"Containers": {
     "qwertyuiopasdfghjklzxcvbnm1234567890": {
                "Name": "lucid_grothendieck",
                "EndpointID": "mnbvcxzlkjhgfdsapoiuytrewq0987654321",
                "MacAddress": "aa:bb:cc:dd:ee:ff",
                "IPv4Address": "172.180.9.3/24",
                "IPv6Address": ""
            },
     ...
}
```

Then you can change triton API url specified as `IPv4Address`.
```python
# httpclient.InferenceServerClient(url="localhost:8000")
httpclient.InferenceServerClient(url="172.180.9.3:8000")
```

## Reference