#include <algorithm>
#include <random>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <nvToolsExt.h>
// Including that causes double free corruption error
//#include <thrust/device_vector.h>
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

#define NUM_INFERENCE 32 


#define cudaCheckError(myerror) {checkGPU((myerror), __FILE__, __LINE__);}

inline void checkGPU(cudaError_t code, const char *file, int line) {

    if (code != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(code) << " in file " << file << ", line " << line << std::endl;
        exit(EXIT_FAILURE);
    }

}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  nvtxRangePushA("Model load");
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::cout << argv[1] << std::endl;
      	  module = torch::jit::load(argv[1], torch::kCUDA);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  nvtxRangePop();
  std::cout << "ok\n";

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);

  // 1's for testing
  float *array = new float[1 * 3 * 224 * 224];
  std::fill_n(array, 1 * 3 * 224 * 224, 1.0f);

  // Random numbers
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(0.0f,255.0f);

  float *input_d;
  cudaCheckError(cudaMalloc((void**)&input_d, 1 * 3 * 224 * 224 * sizeof(float)));
  //cudaCheckError(cudaFree(input_d));
  
  for (int iinf = 0; iinf < NUM_INFERENCE; ++iinf) {
  
    nvtxRangePushA("Generating random numbers on host");
    std::vector<float> input(1 * 3 * 224 * 224);
    std::generate(input.begin(), input.end(), [&dist, &engine]() { return dist(engine); });
    std::generate(input.begin(), input.end(), [&dist, &engine]() { return 1.0f; });
    //cudaMemcpy(input_d, &input[0], 1 * 3 * 224 * 224 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(input_d, array, 1 * 3 * 224 * 224 * sizeof(float), cudaMemcpyHostToDevice);
    //thrust::device_vector<float> input_d = input;
    nvtxRangePop();

    nvtxRangePushA("Loading input");
    std::vector<torch::jit::IValue> inputs;
    //torch::Tensor tensor = torch::ones({1, 3, 224, 224}).to(torch::kCUDA);
    // That doesn't work
    //torch::Tensor tensor = torch::from_blob(array, {1, 3, 224, 224}, options).clone();
    
    // That uses a standard C++ array as a data source to the tensor
    // This array is filled with 1's
    //torch::Tensor tensor = torch::from_blob(array, {1, 3, 224, 224});
    
    // This uses a standard C++ vector as a data source to the tensor
    // This tensor is filled with random numbers
    //torch::Tensor tensor = torch::from_blob(input.data(), {1, 3, 224, 224}).clone();
    // Both above cases need the conversion to the device
    //inputs.push_back(tensor.to(torch::kCUDA));
 
    // This weirdly works even with the above double free corruption error   
    // This uses a Thrust device vector as a data source to the tensor
    torch::Tensor tensor = torch::from_blob(input_d, {1, 3, 224, 224}, options).clone();
    // Already on device
    inputs.push_back(tensor);
    nvtxRangePop();

    // Execute the model and turn its output into a tensor.
    nvtxRangePushA("Model execution");
    //at::Tensor output = module.forward(inputs).toTensor();
    at::Tensor output = module.forward(inputs).toTensor();
    nvtxRangePop();
    std::cout << output.slice(1, 0, 5) << '\n';

    //cudaFree(input_d);
  }
  
  delete [] array;
}


