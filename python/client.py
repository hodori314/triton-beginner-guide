import json
import numpy as np
import tritonclient.http as httpclient

# triton client
client = httpclient.InferenceServerClient(url="localhost:8000")

# read input data
with open("inference_sample.json", "r") as file:
    json_data = json.load(file)

input_data = json_data['data']
input_data = np.array(input_data, dtype=np.float32)

# create input object and output object
detection_input = httpclient.InferInput("input_0", [1, 5], datatype="FP32")
detection_input.set_data_from_numpy(input_data, binary_data=False)

detection_output = httpclient.InferRequestedOutput("output_0", binary_data=False)

# query the server
detection_response = client.infer(model_name="elastic_example", inputs=[detection_input], outputs=[detection_output])
result = detection_response.get_response()
print(result, "\n")

output0 = result["outputs"]
print("Regression Result:", output0, "\n")

# statistics
statistics = client.get_inference_statistics(model_name="elastic_example")
print(statistics)