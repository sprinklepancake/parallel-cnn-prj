#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include <cuda.h>
#include <cuda_runtime.h>
#include "mnist.h"
#include <device_launch_parameters.h>

#include "layer_c.h"


#include <cstdio>
#include <time.h>

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

//define layers of CNN
static Layer layer_input = Layer(0, 0, 28*28);
static Layer layer_c1 = Layer(5*5, 6, 24*24*6);
static Layer layer_s1 = Layer(4*4, 1, 6*6*6);
static Layer layer_f = Layer(6*6*6, 10, 10);

static void learn();
static unsigned int classify(double data[28][28]);
static void test();
static double forward_pass(double data[28][28]);
static double back_pass();

struct KernelConfig {
    dim3 blocks;
    dim3 threads;
};

static inline void loaddata()
{
	mnist_load("/content/Parallel-CNN/data/train-images.idx3-ubyte", "/content/Parallel-CNN/data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("/content/Parallel-CNN/data/t10k-images.idx3-ubyte", "/content/Parallel-CNN/data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}
int main(int argc, const char** argv) {
  srand(time(NULL)); //get rand nmb to start

  //initialize CUDA
  CUresult err = cuInit(0);
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "CUDA initialization failed with error code - %d\n", err);
    return 1;
  }

  loaddata();

  learn();

  test();

  cudaDeviceSynchronize(); //reset for resources management

  return 0;
}
// Forward propagation of a single row in dataset
static double forward_pass(double data[28][28])
{
	float input[28][28];

	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			input[i][j] = data[i][j];
		}
	}

	layer_input.clear();
	layer_c1.clear();
	layer_s1.clear();
	layer_f.clear();

		clock_t start, end;
	start = clock();

	layer_input.setOutput((float *)input);
	KernelConfig configLayer1 = {dim3(6), dim3(24, 24)};
 
  forwP_c1<<<configLayer1.blocks, configLayer1.threads>>>((float (*)[28])layer_input.output, (float (*)[24][24])layer_c1.preact, (float (*)[5][5])layer_c1.weight,layer_c1.bias);
	  apply_step_function<<<configLayer1.blocks, configLayer1.threads>>>(layer_c1.preact, layer_c1.output, layer_c1.O);
   


		  // Pooling layer

		// Configuration for the subsampling layer
KernelConfig configSubsample1 = {
    dim3((6 + 2 - 1) / 2, (6 + 2 - 1) / 2, 6), // Grid size, rounding up if not a perfect multiple
    dim3(2, 2, 1)  // Block size
};
	KernelConfig configBiasS1 = {
    dim3(2, 2, 2), // Blocks
    dim3(3, 3, 3)  // Threads per block
};

	forwP_s1<<<configSubsample1.blocks, configSubsample1.threads>>>((float (*)[24][24])layer_c1.output, (float (*)[6][6])layer_s1.preact, (float (*)[4][4])layer_s1.weight,layer_s1.bias);

	apply_step_function<<<configSubsample1.blocks, configSubsample1.threads>>>(layer_s1.preact, layer_s1.output, layer_s1.O);
  
  

		 // Fully connected layer

	  KernelConfig configFullyConnected = {dim3(10), dim3(256)};

forwP_f<<<configFullyConnected.blocks, configFullyConnected.threads>>>((float (*)[6][6])layer_s1.output, layer_f.preact, (float (*)[6][6][6])layer_f.weight,layer_f.bias);
apply_step_function<<<1, 10>>>(layer_f.preact, layer_f.output, layer_f.O);
  
	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;

}

// Back propagation to update weights
static double back_pass()
{
clock_t start, end;

	start = clock();

int blockSize = 256;  // Optimal block size
int numOutputs = 10;
int gridSize = (numOutputs + blockSize - 1) / blockSize;
	
backP_f<<<gridSize, blockSize>>>((float (*)[6][6][6])layer_f.d_weight,layer_f.bias, layer_f.d_preact, (float (*)[6][6])layer_s1.output);
 
  backP_output_s1<<<5,(216 + 5 - 1) / 5>>>((float (*)[6][6])layer_s1.d_output, (float (*)[6][6][6])layer_f.weight, layer_f.d_preact);
	dim3 threadsPerBlock_s1(6, 6, 6); // One thread for each element in the 6x6x6 block
dim3 numBlocks_s1(1, 1, 1);
	backP_preact_s1<<<numBlocks_s1, threadsPerBlock_s1>>>((float (*)[6][6])layer_s1.d_preact, (float (*)[6][6])layer_s1.d_output, (float (*)[6][6])layer_s1.preact);
	dim3 threadsPerBlock_w_s1(4, 4); // Perfect fit for 4x4 kernel weight dimensions
dim3 numBlocks_w_s1(1, 1);
	backP_weight_s1<<<numBlocks_w_s1, threadsPerBlock_w_s1>>>((float (*)[4][4])layer_s1.d_weight, (float (*)[6][6])layer_s1.d_preact, (float (*)[24][24])layer_c1.output);
	int totalThreads=6*6*6;
	int numBlocks = (totalThreads + 256 - 1);
	backP_bias_s1<<<numBlocks, 256>>>(layer_s1.bias, (float (*)[6][6])layer_s1.d_preact);
    
		
dim3 threadsPerBlock_output_c1(8,8 );  // 4x4 threads to handle the 4x4 weight matrix
dim3 numBlocks_output_c1((24 + threadsPerBlock_output_c1.x - 1) / threadsPerBlock_output_c1.x,
               (24 + threadsPerBlock_output_c1.y - 1) / threadsPerBlock_output_c1.y,
               6);
					
	backP_output_c1<<<numBlocks_output_c1, threadsPerBlock_output_c1>>>((float (*)[24][24])layer_c1.d_output, (float (*)[4][4])layer_s1.weight, (float (*)[6][6])layer_s1.d_preact);
	
	dim3 threadsPerBlock_backP_preact_c1(8, 8); // This can be tuned based on the device capabilities
dim3 numBlocks_backP_preact_c1(
    (24 + threadsPerBlock_backP_preact_c1.x - 1) / threadsPerBlock_backP_preact_c1.x,
    (24 + threadsPerBlock_backP_preact_c1.y - 1) / threadsPerBlock_backP_preact_c1.y,
    6
);
  backP_preact_c1<<<numBlocks_backP_preact_c1, threadsPerBlock_backP_preact_c1>>>((float (*)[24][24])layer_c1.d_preact, (float (*)[24][24])layer_c1.d_output, (float (*)[24][24])layer_c1.preact);
	dim3 threadsPerBlock_weight_c1(5, 5); // Assuming the kernel size is small enough to fit a block
dim3 numBlocks_weight_c1(1, 1, 6); 
	backP_weight_c1<<<numBlocks_weight_c1, threadsPerBlock_weight_c1>>>((float (*)[5][5])layer_c1.d_weight, (float (*)[24][24])layer_c1.d_preact, (float (*)[28])layer_input.output);
	dim3 blocks_bias_c1(6); // One block per feature map
dim3 threads_bias_c1(16, 16);
	backP_bias_c1<<<blocks_bias_c1, threads_bias_c1>>>(layer_c1.bias, (float (*)[24][24])layer_c1.d_preact);
	apply_grad<<<64, 64>>>(layer_f.weight, layer_f.d_weight, layer_f.M * layer_f.N);
	apply_grad<<<64, 64>>>(layer_s1.weight, layer_s1.d_weight, layer_s1.M * layer_s1.N);
	apply_grad<<<64, 64>>>(layer_c1.weight, layer_c1.d_weight, layer_c1.M * layer_c1.N);
	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;

}

static void learn()
{
	static cublasHandle_t blas;
	cublasCreate(&blas);

	float err;
	int iter = 1;
	
	double time_taken = 0.0;

	fprintf(stdout ,"Learning\n");

	while (iter < 0 || iter-- > 0) {
		err = 0.0f;

		for (int i = 0; i < train_cnt; ++i) {
			float tmp_err;

			time_taken += forward_pass(train_set[i].data);

			layer_f.backP_clear();
			layer_s1.backP_clear();
			layer_c1.backP_clear();

			// Euclid distance of train_set[i]
			makeError<<<10, 1>>>(layer_f.d_preact, layer_f.output, train_set[i].label, 10);
			cublasSnrm2(blas, 10, layer_f.d_preact, 1, &tmp_err);
			err += tmp_err;

			time_taken += back_pass();
		}

		err /= train_cnt;
		fprintf(stdout, "error: %e, time_on_gpu: %lf\n", err, time_taken);

		if (err < threshold) {
			fprintf(stdout, "Training complete, error less than threshold\n\n");
			break;
		}

	}
	
	fprintf(stdout, "\n Time - %lf\n", time_taken);
}


// Returns label of given data (0-9)
static unsigned int classify(double data[28][28])
{
	float res[10];

	forward_pass(data);

	unsigned int max = 0;

	cudaMemcpy(res, layer_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

	return max;
}

// Perform forward propagation of test data
static void test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		if (classify(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}

	fprintf(stdout, "Error Rate: %.2lf%%\n",
		double(error) / double(test_cnt) * 100.0);
}