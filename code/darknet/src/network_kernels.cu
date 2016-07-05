#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "params.h"
#include "parser.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "detection_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "deconvolutional_layer.h"
#include "maxpool_layer.h"
#include "avgpool_layer.h"
#include "normalization_layer.h"
#include "cost_layer.h"
#include "local_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "blas.h"

#include "gemm.h"

#include <sys/time.h>
inline double timing(){
        double time;
        struct timeval timmer;

        gettimeofday(&timmer,NULL);
        time = timmer.tv_sec*1e3 + timmer.tv_usec*1e-3;        
        return time;
}





}

float * get_network_output_gpu_layer(network net, int i);
float * get_network_delta_gpu_layer(network net, int i);
float * get_network_output_gpu(network net);

void forward_network_gpu(network net, network_state state)
{
	clock_t time;
	clock_t time1;
	double time1_start = timing();
	double time_start;
    int i;
    for(i = 0; i < net.n; ++i){
        state.index = i;
        layer l = net.layers[i];
        if(l.delta_gpu){
            fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        if(l.type == CONVOLUTIONAL){
/*	
	if(i == 1)
	{	time = clock();
		forward_convolutional_layer_gpu_pre(l, state);
		printf("convolutional_layer_gpu is %f seconds.\n",sec(clock()-time));
	}
*/
	time_start = timing();
           // forward_convolutional_layer_gpu(l, state);
	    forward_convolutional_layer_gpu(l, state,i);
	printf("convolutional_layer_gpu is %f ms.%i.\n",(timing()-time_start),i);
        } else if(l.type == DECONVOLUTIONAL){
	time_start = timing();
            forward_deconvolutional_layer_gpu(l, state);
	printf("forward_deconvolutional_layer_gpu is %f ms.\n",(timing()-time_start));
        } else if(l.type == ACTIVE){
	time_start = timing(); 
            forward_activation_layer_gpu(l, state);
	printf("forward_activation_layer_gpu is %f ms.\n",(timing()-time_start));
        } else if(l.type == LOCAL){
	time = timing(); 
            forward_local_layer_gpu(l, state);
	printf("forward_activation_layer_gpu is %f ms.\n",(timing()-time));
        } else if(l.type == DETECTION){
	time_start = timing(); 
            forward_detection_layer_gpu(l, state);
	printf("forward_detection_layer_gpu is %f ms.\n",(timing()-time_start));
        } else if(l.type == CONNECTED){
	time_start = timing();
            forward_connected_layer_gpu(l, state);
	printf("forward_connected_layer_gpu is %f ms.%i.\n",(timing()-time_start),i);
        } else if(l.type == CROP){
	time_start = timing();
            forward_crop_layer_gpu(l, state);
	printf("forward_crop_layer_gpu is %f ms.\n",(timing()-time_start));
        } else if(l.type == COST){
	time_start = timing();
            forward_cost_layer_gpu(l, state);
	printf("forward_cost_layer_gpu is %f ms.\n",(timing()-time_start));
        } else if(l.type == SOFTMAX){
	time_start = timing();
            forward_softmax_layer_gpu(l, state);
	printf("forward_softmax_layer_gpu is %f ms.\n",(timing()-time_start));
        } else if(l.type == NORMALIZATION){
	time_start = timing();
            forward_normalization_layer_gpu(l, state);
	printf("forward_normalization_layer_gpu is %f ms.\n",(timing()-time_start));
        } else if(l.type == MAXPOOL){
	time_start = timing();
            forward_maxpool_layer_gpu(l, state);
	printf("forward_maxpool_layer_gpu is %f ms.\n",(timing()-time_start));
        } else if(l.type == AVGPOOL){
	time_start = timing();
            forward_avgpool_layer_gpu(l, state);
	printf("forward_avgpool_layer_gpu is %f ms.\n",(timing()-time_start));
        } else if(l.type == DROPOUT){
	time_start = timing();
            forward_dropout_layer_gpu(l, state);
	printf("forward_activation_layer_gpu is %f ms.\n",(timing()-time_start));
        } else if(l.type == ROUTE){
	time_start = timing();
            forward_route_layer_gpu(l, net);
	printf("forward_route_layer_gpu is %f ms.\n",(timing()-time_start));
        } else if(l.type == SHORTCUT){
	time_start = timing();
            forward_shortcut_layer_gpu(l, state);
	printf("forward_shortcut_layer_gpu is %f ms.\n",(timing()-time_start));
        }
        state.input = l.output_gpu;
    }
	printf("tiem1 is %f ms.\n",(timing()-time1_start));
}

void backward_network_gpu(network net, network_state state)
{
    int i;
    float * original_input = state.input;
    float * original_delta = state.delta;
    for(i = net.n-1; i >= 0; --i){
        state.index = i;
        layer l = net.layers[i];
        if(i == 0){
            state.input = original_input;
            state.delta = original_delta;
        }else{
            layer prev = net.layers[i-1];
            state.input = prev.output_gpu;
            state.delta = prev.delta_gpu;
        }
        if(l.type == CONVOLUTIONAL){
            backward_convolutional_layer_gpu(l, state);
        } else if(l.type == DECONVOLUTIONAL){
            backward_deconvolutional_layer_gpu(l, state);
        } else if(l.type == ACTIVE){
            backward_activation_layer_gpu(l, state);
        } else if(l.type == LOCAL){
            backward_local_layer_gpu(l, state);
        } else if(l.type == MAXPOOL){
            if(i != 0) backward_maxpool_layer_gpu(l, state);
        } else if(l.type == AVGPOOL){
            if(i != 0) backward_avgpool_layer_gpu(l, state);
        } else if(l.type == DROPOUT){
            backward_dropout_layer_gpu(l, state);
        } else if(l.type == DETECTION){
            backward_detection_layer_gpu(l, state);
        } else if(l.type == NORMALIZATION){
            backward_normalization_layer_gpu(l, state);
        } else if(l.type == SOFTMAX){
            if(i != 0) backward_softmax_layer_gpu(l, state);
        } else if(l.type == CONNECTED){
            backward_connected_layer_gpu(l, state);
        } else if(l.type == COST){
            backward_cost_layer_gpu(l, state);
        } else if(l.type == ROUTE){
            backward_route_layer_gpu(l, net);
        } else if(l.type == SHORTCUT){
            backward_shortcut_layer_gpu(l, state);
        }
    }
}

void update_network_gpu(network net)
{
    int i;
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            update_convolutional_layer_gpu(l, update_batch, rate, net.momentum, net.decay);
        } else if(l.type == DECONVOLUTIONAL){
            update_deconvolutional_layer_gpu(l, rate, net.momentum, net.decay);
        } else if(l.type == CONNECTED){
            update_connected_layer_gpu(l, update_batch, rate, net.momentum, net.decay);
        } else if(l.type == LOCAL){
            update_local_layer_gpu(l, update_batch, rate, net.momentum, net.decay);
        }
    }
}

float train_network_datum_gpu(network net, float *x, float *y)
{
    network_state state;
    state.index = 0;
    state.net = net;
    int x_size = get_network_input_size(net)*net.batch;
    int y_size = get_network_output_size(net)*net.batch;
    if(net.layers[net.n-1].type == DETECTION) y_size = net.layers[net.n-1].truths*net.batch;
    if(!*net.input_gpu){
        *net.input_gpu = cuda_make_array(x, x_size);
        *net.truth_gpu = cuda_make_array(y, y_size);
    }else{
        cuda_push_array(*net.input_gpu, x, x_size);
        cuda_push_array(*net.truth_gpu, y, y_size);
    }
    state.input = *net.input_gpu;
    state.delta = 0;
    state.truth = *net.truth_gpu;
    state.train = 1;
    forward_network_gpu(net, state);
    backward_network_gpu(net, state);
    float error = get_network_cost(net);
    if (((*net.seen) / net.batch) % net.subdivisions == 0) update_network_gpu(net);

    return error;
}

float *get_network_output_layer_gpu(network net, int i)
{
    layer l = net.layers[i];
    cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
    return l.output;
}

float *get_network_output_gpu(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return get_network_output_layer_gpu(net, i);
}

float *network_predict_gpu(network net, float *input)
{
    int size = get_network_input_size(net) * net.batch;
    network_state state;
    state.index = 0;
    state.net = net;
    state.input = cuda_make_array(input, size);
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
	
/*
int m = 1;
	int k = 1;
	int n = 1;
	float *a;
	float *b;
	float *c;
	float d,e,f;
	d = e = f = 1.0;
	a = &d;
	b = &e;
	c = &f;
	gemm_ongpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

*/
//clock_t time;
//time = clock();
    forward_network_gpu(net, state);
//printf("convolutional_layer_gpu is %f seconds.\n",sec(clock()-time));
    float *out = get_network_output_gpu(net);
    cuda_free(state.input);
    return out;
}

