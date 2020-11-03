#include <cstdlib>
#include <iostream>
#include <memory>

#include <nvToolsExt.h>
#include <torch/script.h> // One-stop header.

#include "H5Cpp.h"

using namespace H5;

#define NUM_INFERENCE 32 
#define IMAGE_DIMS 3
#define CHANNELS 8
#define HEIGHT 125
#define WIDTH 125 
#define IMAGE_SIZE CHANNELS * HEIGHT * WIDTH

int main(int argc, const char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: example-app <path-to-exported-script-module> <path-to-hdf5-file> <batch-size>\n";
    return -1;
  }

  int batch_size = std::atoi(argv[3]);

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

  try {
    std::cout << argv[2] << "\n";
    H5File h5file = H5File(argv[2], H5F_ACC_RDONLY);
    DataSet dataset = h5file.openDataSet("X_jets");

    std::cout << "File " << argv[2] << " read OK..\n";
    H5T_class_t type_class = dataset.getTypeClass();

    FloatType intype = dataset.getFloatType();
    size_t insize = intype.getSize();
    if (type_class == H5T_FLOAT) {
      std::cout << "Data is in FLOAT format, with " << insize << "B per sample\n";
    }

    DataSpace dataspace = dataset.getSpace();
    int rank = dataspace.getSimpleExtentNdims();
    std::cout << "Number of dimensions: " << rank << "\n";

    hsize_t *dims = new hsize_t[rank];
    int ndims = dataspace.getSimpleExtentDims(dims, NULL);

    std::cout << "Dimension sizes: \n";
    for (int idim = 0; idim < rank; ++idim) {
      std::cout << "\t" << idim << ": " << dims[idim] << "\n";
    }
  
  for (int iinf = 0; iinf < int(NUM_INFERENCE / batch_size); ++iinf) {

    std::cout << "Running on image " << iinf << std::endl;

    // Prepare the HDF5 file for reading
    hsize_t offset[2];
    offset[0] = iinf * batch_size;
    offset[1] = 0;
    hsize_t count[2];
    count[0] = batch_size;
    count[1] = IMAGE_SIZE;
    dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
    
    // Prepeare the memory to put the data in
    double *image_data = new double[IMAGE_SIZE * batch_size];
    hsize_t image_dims[2];
    image_dims[0] = batch_size;
    image_dims[1] = IMAGE_SIZE;
    DataSpace image_space(2, image_dims);
    hsize_t m_offset[2];
    m_offset[0] = 0;
    m_offset[1] = 0;
    hsize_t m_count[2];
    m_count[0] = batch_size;
    m_count[1] = IMAGE_SIZE;
    image_space.selectHyperslab(H5S_SELECT_SET, m_count, m_offset);
    dataset.read(image_data, PredType::NATIVE_DOUBLE, image_space, dataspace);
   
    nvtxRangePushA("Loading input");
    std::vector<torch::jit::IValue> inputs;
   
    // Read the data into the tensor - still in F64 format 
    torch::Tensor init_tensor = torch::from_blob(image_data, {batch_size, IMAGE_SIZE}, torch::dtype(torch::kFloat64));
    // Convert the data into F32 format
    torch::Tensor tensor = init_tensor.to(torch::kFloat32);
    // Push the data to the device
    inputs.push_back(tensor.to(torch::kCUDA));
    nvtxRangePop(); // Loading input

    // Execute the model and turn its output into a tensor.
    nvtxRangePushA("Model execution");
    at::Tensor output = module.forward(inputs).toTensor();
    nvtxRangePop(); // Model execution
    std::cout << output << "\n\n";
    
    delete [] image_data;
  }
  
  } catch (FileIException error) {
    std::cerr << "Problem reading HDF5 file " << std::endl;
    error.printError();
    return -1;
  }
}


