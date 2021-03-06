#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"

#include <sys/time.h>
inline double timing(){
        double time;
        struct timeval timmer;

        gettimeofday(&timmer,NULL);
        time = timmer.tv_sec*1e3 + timmer.tv_usec*1e-3;        
        return time;
}
}

__global__ void scale_bias_kernel(float *output, float *biases, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if(offset < size) output[(batch*n+filter)*size + offset] *= biases[filter];
}

void scale_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    dim3 dimGrid((size-1)/BLOCK + 1, n, batch);
    dim3 dimBlock(BLOCK, 1, 1);

    scale_bias_kernel<<<dimGrid, dimBlock>>>(output, biases, n, size);
    check_error(cudaPeekAtLastError());
}

__global__ void backward_scale_kernel(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    __shared__ float part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index]*x_norm[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) scale_updates[filter] += part[i];
    }
}

void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    backward_scale_kernel<<<n, BLOCK>>>(x_norm, delta, batch, n, size, scale_updates);
    check_error(cudaPeekAtLastError());
}

__global__ void add_bias_kernel(float *output, float *biases, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if(offset < size) output[(batch*n+filter)*size + offset] += biases[filter];
}

void add_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    dim3 dimGrid((size-1)/BLOCK + 1, n, batch);
    dim3 dimBlock(BLOCK, 1, 1);

    add_bias_kernel<<<dimGrid, dimBlock>>>(output, biases, n, size);
    check_error(cudaPeekAtLastError());
}

__global__ void backward_bias_kernel(float *bias_updates, float *delta, int batch, int n, int size)
{
    __shared__ float part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) bias_updates[filter] += part[i];
    }
}

void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size)
{
    backward_bias_kernel<<<n, BLOCK>>>(bias_updates, delta, batch, n, size);
    check_error(cudaPeekAtLastError());
}

void forward_convolutional_layer_gpu(convolutional_layer l, network_state state,int ln)
{
    int n = convolutional_out_height(l)*
        convolutional_out_width(l);
	int m = l.n;
fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);


#define Opti1

#ifdef BASE

//if(ln == 1||ln == 3||ln == 5||ln == 6||ln == 7){
//if(ln == 1){
    int i;
    //int m = l.n;
    int k = l.size*l.size*l.c;
    //int n = convolutional_out_height(l)*convolutional_out_width(l);

    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    for(i = 0; i < l.batch; ++i){
        im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, l.col_image_gpu);
        float * a = l.filters_gpu;
        float * b = l.col_image_gpu;
        float * c = l.output_gpu;
        
        double Ops;
        int out_h = convolutional_out_height(l);
    	int out_w = convolutional_out_width(l);
    	int lcc = l.c;
    	int kernel = l.size;
    	Ops = 2*m*out_h*out_w*(((double)(lcc*kernel*kernel))/1000000.0);
    	cudaDeviceSynchronize();
		double start = timing();
		int item;
	int itemN=100;
	    for(item=0;item<itemN;item++)
        	gemm_ongpu(0,0,m,n,k,1.,a,k,b,n,1.,c+i*m*n,n);
        cudaDeviceSynchronize();
        double convtime = (timing()-start)/itemN;
        printf("|----convolution operations time is %f ms,performance is %f GFLOPS for %dX%d * %dX%d \n",convtime*itemN,(Ops/convtime), l.n,l.size*l.size*l.c,l.size*l.size*l.c,out_h*out_w);
	 // printf("absdsdfasdfasdfasdfasf\n");
    }

//}

#endif


#ifdef Opti1

//else{
///*
//added by fanghao
    float *a = l.filters_gpu;
    float *c = l.output_gpu;
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    //int m = l.n;
 
    int lcc = l.c;
    int lhh = l.h;
    int lww = l.w;
    int kernel = l.size;
    int pad;
    if(l.pad)
         pad = l.size/2;
    else
     pad = l.pad;
    lhh += 2*pad;
    lww += 2*pad;
    float *dataP;
    //float *dataP_gpu;
    //dataP = (float *)calloc(lcc*lhh*lww, sizeof(float));
//printf("the k is %i\n",2*l.n*l.size*l.size*n*l.c);
	//printf("long is %i\n",out_w*m);
    dataP = cuda_make_array(NULL, lcc*lhh*lww);
    padding_ongpu(state.input, lcc, lhh, lww, pad, dataP);
//printf("out_h is %i\n",out_h);

    double Ops;
    Ops = 2*m*out_h*out_w*(lcc*kernel*kernel/1000000.0);
    cudaDeviceSynchronize();
    double start = timing();
    int item;
    int itemN=1;
    for(item=0;item<itemN;item++){
        convolutional_ongpu(lhh,lww,m,out_h,out_w,lcc,kernel,a,dataP,l.stride,c);
        cudaDeviceSynchronize();
    }
    double convtime = (timing()-start)/itemN;
    printf("|----convolution operations time is %f ms,performance is %f GFLOPS\n",convtime*itemN,Ops/convtime);
    //cuda_free(dataP);
//*/

//}

#endif

    if(l.batch_normalize){
        if(state.train){
            fast_mean_gpu(l.output_gpu, l.batch, l.n, l.out_h*l.out_w, l.mean_gpu);
            fast_variance_gpu(l.output_gpu, l.mean_gpu, l.batch, l.n, l.out_h*l.out_w, l.variance_gpu);

            /*
            cuda_pull_array(l.variance_gpu, l.mean, 1);
            printf("%f\n", l.mean[0]);
            */


            scal_ongpu(l.n, .95, l.rolling_mean_gpu, 1);
            axpy_ongpu(l.n, .05, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
            scal_ongpu(l.n, .95, l.rolling_variance_gpu, 1);
            axpy_ongpu(l.n, .05, l.variance_gpu, 1, l.rolling_variance_gpu, 1);

            copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
            normalize_gpu(l.output_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.n, l.out_h*l.out_w);
            copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_norm_gpu, 1);
        } else {
            normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.n, l.out_h*l.out_w);
        }

        scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.n, l.out_h*l.out_w);
    }
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, n);

    activate_array_ongpu(l.output_gpu, m*n*l.batch, l.activation);
}


void forward_convolutional_layer_gpu_pre(convolutional_layer l, network_state state)
{
    int n = convolutional_out_height(l)*
        convolutional_out_width(l);
//fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);




    int i;
    int m = l.n;
    int k = l.size*l.size*l.c;
    //int n = convolutional_out_height(l)*convolutional_out_width(l);

    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    for(i = 0; i < l.batch; ++i){
        im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, l.col_image_gpu);
        float * a = l.filters_gpu;
        float * b = l.col_image_gpu;
        float * c = l.output_gpu;
        gemm_ongpu(0,0,m,n,k,1.,a,k,b,n,1.,c+i*m*n,n);
	 // printf("absdsdfasdfasdfasdfasf\n");
    }


    if(l.batch_normalize){
        if(state.train){
            fast_mean_gpu(l.output_gpu, l.batch, l.n, l.out_h*l.out_w, l.mean_gpu);
            fast_variance_gpu(l.output_gpu, l.mean_gpu, l.batch, l.n, l.out_h*l.out_w, l.variance_gpu);

            /*
            cuda_pull_array(l.variance_gpu, l.mean, 1);
            printf("%f\n", l.mean[0]);
            */


            scal_ongpu(l.n, .95, l.rolling_mean_gpu, 1);
            axpy_ongpu(l.n, .05, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
            scal_ongpu(l.n, .95, l.rolling_variance_gpu, 1);
            axpy_ongpu(l.n, .05, l.variance_gpu, 1, l.rolling_variance_gpu, 1);

            copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
            normalize_gpu(l.output_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.n, l.out_h*l.out_w);
            copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_norm_gpu, 1);
        } else {
            normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.n, l.out_h*l.out_w);
        }

        scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.n, l.out_h*l.out_w);
    }
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, n);

    activate_array_ongpu(l.output_gpu, m*n*l.batch, l.activation);
}








void backward_convolutional_layer_gpu(convolutional_layer l, network_state state)
{
    int i;
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = convolutional_out_height(l)*
        convolutional_out_width(l);

    gradient_array_ongpu(l.output_gpu, m*k*l.batch, l.activation, l.delta_gpu);

    backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, k);

    if(l.batch_normalize){
        backward_scale_gpu(l.x_norm_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h, l.scale_updates_gpu);

        scale_bias_gpu(l.delta_gpu, l.scales_gpu, l.batch, l.n, l.out_h*l.out_w);

        fast_mean_delta_gpu(l.delta_gpu, l.variance_gpu, l.batch, l.n, l.out_w*l.out_h, l.mean_delta_gpu);
        fast_variance_delta_gpu(l.x_gpu, l.delta_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.n, l.out_w*l.out_h, l.variance_delta_gpu);
        normalize_delta_gpu(l.x_gpu, l.mean_gpu, l.variance_gpu, l.mean_delta_gpu, l.variance_delta_gpu, l.batch, l.n, l.out_w*l.out_h, l.delta_gpu);
    }

    for(i = 0; i < l.batch; ++i){
        float * a = l.delta_gpu;
        float * b = l.col_image_gpu;
        float * c = l.filter_updates_gpu;

        im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, l.col_image_gpu);
        gemm_ongpu(0,1,m,n,k,1,a + i*m*k,k,b,k,1,c,n);

        if(state.delta){
            float * a = l.filters_gpu;
            float * b = l.delta_gpu;
            float * c = l.col_image_gpu;

            gemm_ongpu(1,0,n,k,m,1,a,n,b + i*k*m,k,0,c,k);

            col2im_ongpu(l.col_image_gpu, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.delta + i*l.c*l.h*l.w);
        }
    }
}

void pull_convolutional_layer(convolutional_layer layer)
{
    cuda_pull_array(layer.filters_gpu, layer.filters, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.filter_updates_gpu, layer.filter_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_pull_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_pull_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_pull_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
}

void push_convolutional_layer(convolutional_layer layer)
{
    cuda_push_array(layer.filters_gpu, layer.filters, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.filter_updates_gpu, layer.filter_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_push_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_push_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_push_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
}

void update_convolutional_layer_gpu(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay)
{
    int size = layer.size*layer.size*layer.c*layer.n;

    axpy_ongpu(layer.n, learning_rate/batch, layer.bias_updates_gpu, 1, layer.biases_gpu, 1);
    scal_ongpu(layer.n, momentum, layer.bias_updates_gpu, 1);

    axpy_ongpu(layer.n, learning_rate/batch, layer.scale_updates_gpu, 1, layer.scales_gpu, 1);
    scal_ongpu(layer.n, momentum, layer.scale_updates_gpu, 1);

    axpy_ongpu(size, -decay*batch, layer.filters_gpu, 1, layer.filter_updates_gpu, 1);
    axpy_ongpu(size, learning_rate/batch, layer.filter_updates_gpu, 1, layer.filters_gpu, 1);
    scal_ongpu(size, momentum, layer.filter_updates_gpu, 1);
}


