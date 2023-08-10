// Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <getopt.h>
#include <unistd.h>

#include <iostream>
#include <string>

#include "http_client.h"

namespace tc = triton::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    tc::Error err = (X);                                           \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

int
main(int argc, char** argv)
{
  /*
   *    HTTP configurations 
   */
  bool verbose = false;
  std::string url("172.180.9.3:8000"); // FIXME: you have to manually change IPAddress to your triton server IP
  tc::Headers http_headers;
  uint32_t client_timeout = 100;
  
  auto request_compression_algorithm = tc::InferenceServerHttpClient::CompressionType::NONE;  /* default */
  auto response_compression_algorithm = tc::InferenceServerHttpClient::CompressionType::NONE;  /* default */

  std::string model_name = "opSubgraph";
  std::string model_version = "1";

  tc::HttpSslOptions ssl_options;  /* default */
  ssl_options.verify_peer = 1; 
  ssl_options.verify_host = 2;
  ssl_options.ca_info = tc::HttpSslOptions::CERTTYPE::CERT_PEM;
  ssl_options.cert = tc::HttpSslOptions::KEYTYPE::KEY_PEM;
  ssl_options.key = "";


  /*
   *    Create a InferenceServerHttpClient instance
   *    to communicate with the server using HTTP protocol
   */
  std::unique_ptr<tc::InferenceServerHttpClient> client;
  FAIL_IF_ERR(tc::InferenceServerHttpClient::Create(&client, url, verbose, ssl_options),
              "unable to create http client");

  /*
   *    Data Generation (In Orca, we does not need)
   */
  std::vector<float> input0_data(27);
  for (size_t i = 0; i < 27; ++i) 
  {
    input0_data[i] = 0.1;
  }
  std::vector<int64_t> shape{1, 27};

  /*
   *    Initialize the inputs with the data, output with empty, results
   *    Empty output vector will request data for all the output tensors from the server
   */
  tc::InferInput* input0;
  FAIL_IF_ERR(tc::InferInput::Create(&input0, "input_0", shape, "FP32"),
              "unable to get input_0");
  
  std::shared_ptr<tc::InferInput> input0_ptr;
  input0_ptr.reset(input0);

  FAIL_IF_ERR(input0_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&input0_data[0]), 
                                    input0_data.size() * sizeof(float)),
              "unable to set data for input_0");


  std::vector<tc::InferInput*> inputs = {input0_ptr.get()};

  std::vector<const tc::InferRequestedOutput*> outputs = {};
  tc::InferResult* results;

  /*
   *     Inference settings
   */
  tc::InferOptions options(model_name);
  options.model_version_ = model_version;
  options.client_timeout_ = client_timeout;

  /*
   *     Inference
   */
  FAIL_IF_ERR(client->Infer(&results, options, inputs, outputs, http_headers, tc::Parameters(),
                            request_compression_algorithm, response_compression_algorithm),
              "unable to run model");
  std::shared_ptr<tc::InferResult> results_ptr;
  results_ptr.reset(results);

  /*
   *    Get pointers to the result returned
   */
  float* output0_data;
  size_t output0_byte_size;
  FAIL_IF_ERR(results_ptr->RawData("output_0", 
                                   (const uint8_t**)&output0_data, 
                                   &output0_byte_size),
              "unable to get result data for 'output_0'");
  
  if (output0_byte_size != 4) {
    std::cerr << "error: received incorrect byte size for 'output_0': "
              << output0_byte_size << std::endl;
    exit(1);
  }

  // Get full response
  std::cout << *output0_data << std::endl;
  std::cout << results_ptr->DebugString() << std::endl;
  
  std::cout << "PASS : Infer" << std::endl;
  return 0;
}