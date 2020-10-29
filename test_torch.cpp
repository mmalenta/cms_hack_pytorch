#include <nvToolsExt.h>
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

#define NUM_INF 16

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  nvtxRangePushA("Model load");
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1], torch::kCUDA);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  nvtxRangePop();


  std::cout << "ok\n";

  // Create a vector of inputs.
  
  for (int iinf = 0; iinf < NUM_INF; ++iinf) {
  
    nvtxRangePushA("Loading input");
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 3, 224, 224}).to(torch::kCUDA));
    nvtxRangePop();

    // Execute the model and turn its output into a tensor.
    nvtxRangePushA("Model execution");
    at::Tensor output = module.forward(inputs).toTensor();
    nvtxRangePop();
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

  }
}


