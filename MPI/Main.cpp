#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h" 
#include "layer.h"
#include <cstdio>
#include <ctime>
#include <vector>
#include <cmath>
#include <memory>
#include <time.h>
#include <mpi.h>


double total_convolution_time = 0, total_pooling_time = 0, total_fully_connected_time = 0,total_gradient_time=0;

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
static Layer l_input(0, 0, 28*28);
static Layer l_c1(5*5, 6, 24*24*6);
static Layer l_s1(4*4, 1, 6*6*6);
static Layer l_f(6*6*6, 10, 10);

static void learn();
static unsigned int classify(double data[28][28]);
static void test();
static double forward_pass(double data[28][28]);
static double back_pass();

float vectorNorm(float* vec, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

static inline void loaddata()
{
	mnist_load("/content/Parallel-CNN/data/train-images.idx3-ubyte", "/content/Parallel-CNN/data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("/content/Parallel-CNN/data/t10k-images.idx3-ubyte", "/content/Parallel-CNN/data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

int main(int argc, char **argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;

    // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    srand(time(NULL));
    loaddata();

    learn();
    test();
    
    if (MPI_COMM_WORLD == 0) {  // Ensure only the master process prints the results
        printf("Total Convolution Time: %f ms\n", total_convolution_time);
        printf("Total Pooling Time: %f ms\n", total_pooling_time);
        printf("Total Fully Connected Time: %f ms\n", total_fully_connected_time);
        printf("Total Time on applying gradients: %f ms\n", total_gradient_time);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}

static double forward_pass(double data[28][28]) {
  float input[28][28];

	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			input[i][j] = data[i][j];
		}
	}

	l_input.clear();
	l_c1.clear();
	l_s1.clear();
	l_f.clear();
    float milliseconds=0;
	clock_t start, end;
    clock_t start_1, end_1;
    start_1=clock();
	

	l_input.setOutput((float *)input);
	 // forward pass Convolution Layer
    start = clock();
    fp_c1_parallel((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight,l_c1.bias);
    apply_step_function(l_c1.preact, l_c1.output, l_c1.O);
    end = clock();
    milliseconds = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    total_convolution_time += milliseconds;

     // forward pass pooling Layer
    start = clock();
    fp_s1((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight,l_s1.bias);
    apply_step_function(l_s1.preact, l_s1.output, l_s1.O);
    end = clock();
    milliseconds = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    total_pooling_time += milliseconds;

 // forward pass Fully Connected Layer
    start = clock();
    fp_preact_f((float (*)[6][6])l_s1.output, l_f.preact, (float (*)[6][6][6])l_f.weight);
    fp_bias_f(l_f.preact, l_f.bias);
    apply_step_function(l_f.preact, l_f.output, l_f.O);
    end = clock();
    milliseconds = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    total_fully_connected_time += milliseconds;
end_1= clock();
	return ((double) (end_1 - start_1)) / CLOCKS_PER_SEC;
}

static double back_pass() {
    clock_t start,end;
     clock_t start_1,end_1;
     start_1=clock();
   
 float milliseconds=0;
start = clock();
    bp_weight_f((float (*)[6][6][6])l_f.d_weight, l_f.d_preact, (float (*)[6][6])l_s1.output);
    bp_bias_f(l_f.bias, l_f.d_preact);
   end = clock();
    milliseconds = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    total_fully_connected_time += milliseconds;
    start = clock();
    bp_output_s1((float (*)[6][6])l_s1.d_output, (float (*)[6][6][6])l_f.weight, l_f.d_preact);
    bp_preact_s1((float (*)[6][6])l_s1.d_preact, (float (*)[6][6])l_s1.d_output, (float (*)[6][6])l_s1.preact);
    bp_weight_s1((float (*)[4][4])l_s1.d_weight, (float (*)[6][6])l_s1.d_preact, (float (*)[24][24])l_c1.output);
    bp_bias_s1(l_s1.bias, (float (*)[6][6])l_s1.d_preact);
       end = clock();
    milliseconds = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    total_pooling_time += milliseconds;
start = clock();
    bp_output_c1((float (*)[24][24])l_c1.d_output, (float (*)[4][4])l_s1.weight, (float (*)[6][6])l_s1.d_preact);
    bp_preact_c1((float (*)[24][24])l_c1.d_preact, (float (*)[24][24])l_c1.d_output, (float (*)[24][24])l_c1.preact);
    bp_weight_c1((float (*)[5][5])l_c1.d_weight, (float (*)[24][24])l_c1.d_preact, (float (*)[28])l_input.output);
    bp_bias_c1(l_c1.bias, (float (*)[24][24])l_c1.d_preact);
end = clock();
    milliseconds = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    total_convolution_time += milliseconds;
    start = clock();
	apply_grad(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
	apply_grad(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
	apply_grad(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);
    end = clock();
    milliseconds = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    total_gradient_time += milliseconds;
end_1= clock();
	return ((double) (end_1- start_1)) / CLOCKS_PER_SEC;
}

static void learn() {

  int rank, size;

    // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float err;
	int iter = 1;
	
	double time_taken = 0.0;
    if(rank == 0){
	fprintf(stdout ,"Learning\n");
    }
	while (iter < 0 || iter-- > 0) {
		err = 0.0f;

		for (int i = 0; i < train_cnt; ++i) {
			float tmp_err;

			time_taken += forward_pass(train_set[i].data);

			l_f.bp_clear();
			l_s1.bp_clear();
			l_c1.bp_clear();

            // Euclid distance of train_set[i]
    makeError(l_f.d_preact, l_f.output, train_set[i].label, 10);
            tmp_err = vectorNorm(l_f.d_preact, 10);
            err += tmp_err;
           time_taken += back_pass();
        }

        if(rank==0){
        err /= train_cnt;
		fprintf(stdout, "error: %e, time_on_cpu: %lf\n", err, time_taken);

		if (err < threshold) {
			fprintf(stdout, "Training complete, error less than threshold\n\n");
			break;
		}
        }

	}

  if(rank==0){
	
	fprintf(stdout, "\n Time - %lf\n", time_taken);

  }
}

static unsigned int classify(double data[28][28]) {
    float res[10];
    forward_pass(data);
    unsigned int max = 0;
   for (int i = 0; i < 10; i++) {
        res[i] = l_f.output[i];
    }
    for (int i = 1; i < 10; ++i) {
        if (res[max] < res[i]) {
            max = i;
        }
    }

    return max;
}

static void test()
{
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		if (classify(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}
  if(rank==0){
	fprintf(stdout, "Error Rate: %.2lf%%\n",
		double(error) / double(test_cnt) * 100.0);
  }
}