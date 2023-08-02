env:
	conda create -n elastic python=3.9 -y

setup:
	conda install numpy scikit-learn skl2onnx tritonclient[http] -y