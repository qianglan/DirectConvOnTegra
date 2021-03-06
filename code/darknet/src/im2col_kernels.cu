#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "im2col.h"
#include "cuda.h"
#include "stdio.h"
}

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

__global__ void im2col_gpu_kernel(const int n, const float* data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_col) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x){
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        float* data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        const float* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;
                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

//added by fanghao
__global__ void padding_ongpu_kernel(const float* input, const int lcc, const int lhh, const int lww, const int pad, float * dataP){

    int ii,jj,kk;
    int tt=0;
    for(tt=0;tt<lcc*lhh*lww;tt++)
	dataP[tt] = 0.0;
    for(ii=0; ii < lcc; ii++)
        for(jj=pad; jj<lhh-pad; jj++)
             for(kk=pad; kk<lww-pad; kk++)
                dataP[ii*lhh*lww + jj*lww + kk] = input[ii*(lhh - 2*pad)*(lww-2*pad) + (jj - pad)*(lww - 2*pad) + kk-pad];

}

__global__ void convolutional_ongpu_kernel(const int lhh,const int lww,const int m,const int out_h,const int out_w,const int lcc,const int kernel,const float *a,const float *dataP,const int stride,float *c){
   /* int ii,jj,kk,mm,pp,tt;
	//float tempAcc = 0.0;
    for(ii=0; ii<m; ii++)
	
        for(jj=0; jj<out_h; jj++)
		//tempAcc += a[ii*kernel*kernel*lcc];
            for(kk=0; kk<out_w; kk++) {
                float tempAcc = 0.0;
		//tempAcc+=a[ii*lcc*kernel*kernel];
                    for(mm=0; mm<lcc; mm++)
			//tempAcc+=a[ii*lcc*kernel*kernel];
                        for(pp=0; pp<kernel; pp++)
				//tempAcc+=a[ii*lcc*kernel*kernel];
				tempAcc+=dataP[mm*lhh*lww];
                            for(tt=0; tt<kernel; tt++)
                                //tempAcc += a[ii*lcc*kernel*kernel+mm*kernel*kernel+pp*kernel+tt]*dataP[mm*lhh*lww+(stride*jj+pp)*lww+stride*kk+tt];
                                // dataP[0];
				//tempAcc+=a[ii*lcc*kernel*kernel];
                c[ii*out_h*out_w+jj*out_w+kk] = tempAcc;
                        }*/

///*
//opti0
//int ii,jj,kk,mm,pp,tt;
#define  Opt2
#ifdef BASE
int mm,pp,tt;
int ii = threadIdx.x;
int jj = blockIdx.x;
int kk = blockIdx.y;
//int ii = blockIdx.x*blockDim.x+threadIdx.x;
    //for(ii=0; ii<m; ii++)
      // for(jj=0; jj<out_h; jj++)
           //for(kk=0; kk<out_w; kk++) {
                float tempAcc = 0.0;
                    for(mm=0; mm<lcc; mm++)
                        for(pp=0; pp<kernel; pp++)
                            for(tt=0; tt<kernel; tt++)
                                tempAcc += a[ii*lcc*kernel*kernel+mm*kernel*kernel+pp*kernel+tt]*dataP[mm*lhh*lww+(stride*jj+pp)*lww+stride*kk+tt];
                c[ii*out_h*out_w+jj*out_w+kk] = tempAcc;
			//c[kk*out_h*m+jj*m+ii] = tempAcc;
                //if(ii==0)
		//	c[ii] = tempAcc;
				//c[ii*out_h*out_w+jj*out_w+kk] += a[ii*lcc*kernel*kernel+mm*kernel*kernel+pp*kernel+tt]*dataP[mm*lhh*lww+(stride*jj+pp)*lww+stride*kk+tt];
             //           }
#endif

#ifdef Opt1
	int mm,pp,tt;
	int ii = threadIdx.x;
	int jj = blockIdx.x;
	int kk = blockIdx.y;
	float tempAcc = 0.0;
	for(tt=0;tt<kernel;tt++)
	    for (pp=0;pp<kernel;pp++)
		for (mm=0;mm<lcc;mm++)
		    tempAcc +=a[ii*lcc*kernel*kernel+tt*lcc*kernel+pp*lcc+mm]*dataP[(stride*kk+tt)*lcc*lww+(stride*jj+pp)*lcc+mm];
	c[kk*out_h*m+jj*m+ii] = tempAcc;


#endif

#ifdef Opt2

    __shared__ float ACC[11][32];
    
    int ii = blockIdx.x;
    int jj = blockIdx.y;
    int kk = blockIdx.z;
    
    int mm = threadIdx.x;
    int pp;// = threadIdx.y;
    int tt = threadIdx.z;
    int mms = (lcc==3)?lcc:blockDim.x;

    float tempAcc = 0.0;
    int CN = (lcc==3)?lcc:lcc/mms;

    int mi;
    if(mm<mms) {
        for(pp=0;pp<kernel;pp++)
            for(mi=0;mi<CN;mi++)
	        tempAcc +=a[ii*lcc*kernel*kernel+tt*lcc*kernel+pp*lcc+mm*mms+mi]*dataP[(stride*kk+tt)*lcc*lww+(stride*jj+pp)*lcc+mm*mms+mi];
    }

    ACC[tt][mm] = tempAcc;
    if(tt==0){
        int ts=0;
	for(ts=1;ts<blockDim.z;ts++)
	    ACC[0][mm] += ACC[ts][mm];
        for(unsigned int s=mms/2;s>0;s>>=1)
        {
	    if(mm<s)
		ACC[0][mm] += ACC[0][mm+s];
            __syncthreads();
        }
        if(mm==0)
	c[kk*out_h*m+jj*m+ii] = tempAcc;
    }
    

#endif
//*/
/*
//opti1
int kk,mm,pp,tt;
int ii = blockIdx.x;
int jj =threadIdx.x ;
__shared__ float SharedC[5400];
//int ii = blockIdx.x*blockDim.x+threadIdx.x;
    //for(ii=0; ii<m; ii++)
      // for(jj=0; jj<out_h; jj++)
for(kk=0; kk<out_w; kk++)
	SharedC[jj*out_w + kk] = 0.0;
//__syncthreads();
           for(kk=0; kk<out_w; kk++) {              
                    for(mm=0; mm<lcc; mm++)
                        for(pp=0; pp<kernel; pp++)
                            for(tt=0; tt<kernel; tt++)
                                SharedC[jj*out_w + kk] += a[ii*lcc*kernel*kernel+mm*kernel*kernel+pp*kernel+tt]*dataP[mm*lhh*lww+(stride*jj+pp)*lww+stride*kk+tt];
				
			 }

for(kk=0; kk<out_w; kk++)
 c[ii*out_h*out_w+jj*out_w+kk] = SharedC[jj*out_w + kk];
			//__syncthreads();

*/

}

__global__ void gemm_ongpu_kernel(const int M,const int N,const int K,const float *a,const float *b,float *c)
{
	int i,j,k;
	for(i = 0; i < M; i++){
		for(k = 0; k < K; k++){
			for(j = 0; j< N; j++){
			    c[i*N+j] += a[i*K+k]*b[k*N+j];
			    }
		}
	}
	
}

 
void  padding_ongpu(float *input, int lcc, int lhh, int lww, int pad, float *dataP){

    int blockpergrids = 1;
    int threadsperblock = 1;
    cudaDeviceSynchronize();
    padding_ongpu_kernel<<<blockpergrids,threadsperblock>>>(input, lcc, lhh, lww, pad, dataP);
    cudaDeviceSynchronize();
}

void convolutional_ongpu(int lhh,int lww,int m,int out_h,int out_w,int lcc,int kernel,float *a,float *dataP,int stride,float *c){
    //int blockpergrids = m;
    //int threadsperblock = out_h;
    dim3 blockpergrids(out_h,out_w);
    dim3 threadsperblock(m);
    cudaDeviceSynchronize();
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop,device);
    int numBlocks;
    int blockSize=m;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,convolutional_ongpu_kernel,blockSize,0);
    int activeWarps = numBlocks*blockSize/prop.warpSize;
    int maxWarps = prop.maxThreadsPerMultiProcessor/prop.warpSize;
    printf("ActiveWarps is %d, maxWarps is %d numBlocks is %d blockSize is%d \n",activeWarps,maxWarps,numBlocks, blockSize);

    int minGridSize;
    int gridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,(void *)convolutional_ongpu_kernel,0,m*out_h*out_w);
    printf("minGridSize is %d,blockSize is %d \n",minGridSize,blockSize);

    convolutional_ongpu_kernel<<<blockpergrids,threadsperblock>>>(lhh,lww,m,out_h,out_w,lcc,kernel,a,dataP,stride,c);
    //cuda_pull_array(float *x_gpu, float *x, int n)
    cudaDeviceSynchronize();

}


void forward_connected_layer_ongpu(const int M,const int N,const int K,const float *a,const float *b, float *c){
    int blockpergrids = 1;
    int threadsperblock = 1;
    cudaDeviceSynchronize();
    gemm_ongpu_kernel<<<blockpergrids,threadsperblock>>>(M,N,K,a,b,c);
    cudaDeviceSynchronize();
}











void im2col_ongpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, float *data_col){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    pad = pad ? ksize/2 : 0;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(
                num_kernels, im, height, width, ksize, pad,
                stride, height_col,
                width_col, data_col);
    cudaDeviceSynchronize();
}






/*
   __global__ void im2col_pad_kernel(float *im,
   int channels,  int height,  int width,
   int ksize,  int stride, float *data_col)
   {
   int c,h,w;
   int height_col = 1 + (height-1) / stride;
   int width_col = 1 + (width-1) / stride;
   int channels_col = channels * ksize * ksize;

   int pad = ksize/2;

   int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
   int col_size = height_col*width_col*channels_col;
   if (id >= col_size) return;

   int col_index = id;
   w = id % width_col;
   id /= width_col;
   h = id % height_col;
   id /= height_col;
   c = id % channels_col;
   id /= channels_col;

   int w_offset = c % ksize;
   int h_offset = (c / ksize) % ksize;
   int im_channel = c / ksize / ksize;
   int im_row = h_offset + h * stride - pad;
   int im_col = w_offset + w * stride - pad;

   int im_index = im_col + width*(im_row + height*im_channel);
   float val = (im_row < 0 || im_col < 0 || im_row >= height || im_col >= width) ? 0 : im[im_index];

   data_col[col_index] = val;
   }

   __global__ void im2col_nopad_kernel(float *im,
   int channels,  int height,  int width,
   int ksize,  int stride, float *data_col)
   {
   int c,h,w;
   int height_col = (height - ksize) / stride + 1;
   int width_col = (width - ksize) / stride + 1;
   int channels_col = channels * ksize * ksize;

   int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
   int col_size = height_col*width_col*channels_col;
   if (id >= col_size) return;

   int col_index = id;
   w = id % width_col;
   id /= width_col;
   h = id % height_col;
   id /= height_col;
   c = id % channels_col;
   id /= channels_col;

   int w_offset = c % ksize;
   int h_offset = (c / ksize) % ksize;
   int im_channel = c / ksize / ksize;
   int im_row = h_offset + h * stride;
   int im_col = w_offset + w * stride;

   int im_index = im_col + width*(im_row + height*im_channel);
   float val = (im_row < 0 || im_col < 0 || im_row >= height || im_col >= width) ? 0 : im[im_index];

   data_col[col_index] = val;
   }

   extern "C" void im2col_ongpu(float *im,
   int channels,  int height,  int width,
int ksize,  int stride,  int pad, float *data_col)
{

    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;

    if (pad){
        height_col = 1 + (height-1) / stride;
        width_col = 1 + (width-1) / stride;
    }

    size_t n = channels_col*height_col*width_col;

    if(pad)im2col_pad_kernel<<<cuda_gridsize(n),BLOCK>>>(im,  channels, height, width, ksize, stride, data_col);
    else im2col_nopad_kernel<<<cuda_gridsize(n),BLOCK>>>(im,  channels, height, width, ksize, stride, data_col);
    check_error(cudaPeekAtLastError());
}
*/
