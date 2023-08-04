import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Generating synthetic data
np.random.seed(42)
n_samples, n_features = 100, 5
X = np.random.rand(n_samples, n_features)
y = 3*X[:, 0] + 2*X[:, 1] - 5*X[:, 2] + np.random.randn(n_samples)

# Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the ElasticNet model
alpha = 0.1  # Constant that multiplies the sum of L1 and L2 penalties (typically a small value close to 0)
l1_ratio = 0.5  # L1 regularization mixing parameter (0: L2 penalty only, 1: L1 penalty only, 0.5: Half L1 and Half L2)
elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
elastic_net.fit(X_train, y_train)

# Making predictions using the trained model
y_pred = elastic_net.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Save the ElasticNet model to a onnx file
model_path = "elastic_example/1/model.onnx"
initial_type = [('input_0', FloatTensorType([None, n_features]))]
final_type = [('output_0', FloatTensorType([ 1 ]))]
onx = convert_sklearn(elastic_net, initial_types=initial_type, target_opset=15, final_types=final_type)
with open(model_path, "wb") as f:
    f.write(onx.SerializeToString())