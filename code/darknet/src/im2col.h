#ifndef IM2COL_H
#define IM2COL_H

void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);

#ifdef GPU

void im2col_ongpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,float *data_col);
void  padding_ongpu(float *input, int lcc, int lhh, int lww, int pad, float *dataP);
void convolutional_ongpu(int lhh,int lww,int m,int out_h,int out_w,int lcc,int kernel,float *a,float *dataP,int stride,float *c);
void forward_connected_layer_ongpu(const int M,const int N,const int K,const float *a,const float *b,float *c);
#endif
#endif
