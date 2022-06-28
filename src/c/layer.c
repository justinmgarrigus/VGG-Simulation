#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include "layer.h" 
#include "progress.h" 

layer* layer_create(int weight_set_count, ndarray** weights, enum layer_type type, enum layer_activation activation, ndarray* outputs) {
	layer *lr = malloc(sizeof(layer));
	
	lr->weight_set_count = weight_set_count; 
	lr->weights = weights; 
	lr->outputs = outputs; 
	
	switch (type) {
		case layer_type_none: 
			break; 
		
		case layer_type_convolutional: 
			lr->feed = layer_convolutional_feedforward; 
			break; 
		
		case layer_type_max_pooling: 
			lr->feed = layer_max_pooling_feedforward; 
			break; 
			
		case layer_type_flatten: 
			lr->feed = layer_flatten_feedforward; 
			break; 
		
		case layer_type_dense: 
			lr->feed = layer_dense_feedforward; 
			break; 
			
		default: 
			fprintf(stderr, "Unknown layer type specified: %d\n", type); 
			exit(-1); 
	}
	
	switch (activation) {
		case layer_activation_none: 
			break; 
		
		case layer_activation_relu: 
			lr->activation = layer_relu; 
			break; 
			
		case layer_activation_softmax: 
			lr->activation = layer_softmax; 
			break; 
			
		default: 
			fprintf(stderr, "Unknown activation type specified: %d\n", activation); 
			exit(-1); 
	}
	
	return lr; 
}

void layer_free(layer* layer) {
	for (int i = 0; i < layer->weight_set_count; i++)
		ndarray_free(layer->weights[i]);
	free(layer->weights);
	ndarray_free(layer->outputs);
	free(layer);
}

void layer_convolutional_feedforward(layer* input_layer, layer* conv_layer) {
	printf("Conv2D "); ndarray_fprint(input_layer->outputs, stdout);
#ifdef DRAW_PROGRESS
	printf("\033[s"); // Save cursor position 
	printf("\n"); 
	struct progressbar *bar = progressbar_create(10, 3, 2);
#else 
	printf("\n"); 
#endif
	
	int padding[4] = {0, 1, 1, 0};
	ndarray *inputs = ndarray_pad(input_layer->outputs, padding);
	
	ndarray *outputs = conv_layer->outputs; 
	ndarray *kernel = conv_layer->weights[0]; 
	ndarray *bias = conv_layer->weights[1];
	
	int counter = 0; 
	int counter_max = outputs->shape[1] * outputs->shape[2]; 
	for (int x = 0; x < outputs->shape[1]; x++) {
		printf("%d\n", x); 
		for (int y = 0; y < outputs->shape[2]; y++) {
			for (int filter_index = 0; filter_index < outputs->shape[3]; filter_index++) {
				ND_TYPE result = ndarray_get_val_param(bias, filter_index);  
				for (int kernel_x = 0; kernel_x < kernel->shape[0]; kernel_x++) {
					for (int kernel_y = 0; kernel_y < kernel->shape[1]; kernel_y++) {
						for (int channel = 0; channel < inputs->shape[3]; channel++) {
							result += ndarray_get_val_param(kernel, kernel_x, kernel_y, channel, filter_index) * ndarray_get_val_param(inputs, 0, x + kernel_x, y + kernel_y, channel);  
						}
					}
				}
				
				result = conv_layer->activation(&result); 
				if (result < 0) result = 0;
				ndarray_set_val_param(outputs, result, 0, x, y, filter_index); 
			}
			
#ifdef DRAW_PROGRESS
			progressbar_draw(bar, (double)(counter++) / counter_max); 
#endif 
		}
	}	
	free(inputs); 

#ifdef DRAW_PROGRESS
	printf("\033[u"); // Restore cursor position
	printf(" \x1B[36m"); // Color cyan 
	double elapsed_time = (current_time_millis() - bar->time_started) / 1000.0; 
	printf(bar->digit_format, elapsed_time); 
	network_operation_time += elapsed_time; 
	printf(" \x1B[35m(%.2f)", network_operation_time); // Color magenta, total time
	printf("\x1B[0m\n"); // Reset color and newline 
	progressbar_free(bar); 
#endif
	
	ndarray_log(outputs, "c_log.txt"); 
}

void layer_max_pooling_feedforward(layer* input_layer, layer* pool_layer) { 
	printf("MaxPooling "); ndarray_fprint(pool_layer->outputs, stdout);
	ndarray *input = input_layer->outputs; 
	ndarray *output = pool_layer->outputs;

#ifdef DRAW_PROGRESS	
	printf("\033[s"); // Save cursor position 
	printf("\n"); 
	struct progressbar *bar = progressbar_create(10, 3, 2);
#else 
	printf("\n"); 
#endif 
	
	for (int offset_x = 0; offset_x < input->shape[1]; offset_x += 2) {
		for (int offset_y = 0; offset_y < input->shape[2]; offset_y += 2) {
			for (int z = 0; z < input->shape[3]; z++) {
				ND_TYPE max_value = 0; 
				for (int kernel_x = 0; kernel_x < 2; kernel_x++) {
					for (int kernel_y = 0; kernel_y < 2; kernel_y++) {
						ND_TYPE val = ndarray_get_val_param(input, 0, offset_x + kernel_x, offset_y + kernel_y, z); 
						if (val > max_value) 
							max_value = val; 
					}
				}
				ndarray_set_val_param(output, max_value, 0, offset_x / 2, offset_y / 2, z); 
			}
		}
		
#ifdef DRAW_PROGRESS
		progressbar_draw(bar, (double)offset_x / input->shape[1]); 
#endif 
	}
	
#ifdef DRAW_PROGRESS
	printf("\033[u"); // Restore cursor position
	printf(" \x1B[36m"); // Color cyan 
	double elapsed_time = (current_time_millis() - bar->time_started) / 1000.0; 
	printf(bar->digit_format, elapsed_time); 
	network_operation_time += elapsed_time; 
	printf(" \x1B[35m(%.2f)", network_operation_time); // Color magenta, total time
	printf("\x1B[0m\n"); // Reset color and newline 
	progressbar_free(bar);  
#endif
}

void layer_flatten_feedforward(layer* input_layer, layer* flatten_layer) { 
	printf("Flatten "); ndarray_fprint(flatten_layer->outputs, stdout); printf(" "); 
	ndarray *input = input_layer->outputs; 
	ndarray *output = flatten_layer->outputs; 
	
#ifdef DRAW_PROGRESS
	printf("\033[s"); // Save cursor position 
	printf("\n"); 
	struct progressbar *bar = progressbar_create(10, 3, 2);
#else 
	printf("\n"); 
#endif
	
	int *pos = malloc(sizeof(int) * input->dim);
	for (int i = 0; i < input->dim; i++)
		pos[i] = 0;
	
	int index = 0; 
	do {
		ndarray_set_val_param(output, ndarray_get_val_list(input, pos), 0, index++);
	}
	while (ndarray_decimal_count(input->dim, pos, input->shape));
	free(pos); 
	
#ifdef DRAW_PROGRESS
	printf("\033[u"); // Restore cursor position
	printf(" \x1B[36m"); // Color cyan 
	double elapsed_time = (current_time_millis() - bar->time_started) / 1000.0; 
	printf(bar->digit_format, elapsed_time); 
	network_operation_time += elapsed_time; 
	printf(" \x1B[35m(%.2f)", network_operation_time); // Color magenta, total time
	printf("\x1B[0m\n"); // Reset color and newline 
	progressbar_free(bar); 
#endif
}

void layer_dense_feedforward(layer* input_layer, layer* dense_layer) { 
	printf("Dense "); ndarray_fprint(dense_layer->outputs, stdout);
	ndarray *input = input_layer->outputs; 
	ndarray *output = dense_layer->outputs; 
	ndarray *weights = dense_layer->weights[0]; 
	ndarray *biases = dense_layer->weights[1];

#ifdef DRAW_PROGRESS
	printf("\033[s"); // Save cursor position 
	printf("\n"); 
	struct progressbar *bar = progressbar_create(10, 3, 2);
#else 
	printf("\n"); 
#endif 

	ND_TYPE expo_sum = 0; 
	for (int i = 0; i < input->shape[0]; i++) {
		for (int j = 0; j < weights->shape[1]; j++) {
			ND_TYPE result = ndarray_get_val_param(biases, j); 
			for (int k = 0; k < weights->shape[0]; k++)
				result += ndarray_get_val_param(input, i, k) * ndarray_get_val_param(weights, k, j); 
			ndarray_set_val_param(output, result, i, j);
			expo_sum += exp(result);
			
#ifdef DRAW_PROGRESS
			progressbar_draw(bar, (double)j / weights->shape[1] * 0.8); // TODO: This assumes input->shape[0] == 1
#endif 			
		}
	}
	
	ND_TYPE acti[2] = { 0, expo_sum }; 
	for (int i = 0; i < input->shape[0]; i++) {
		for (int j = 0; j < weights->shape[1]; j++) {
			acti[0] = ndarray_get_val_param(output, i, j); 
			ndarray_set_val_param(output, dense_layer->activation(acti), i, j); 
			
#ifdef DRAW_PROGRESS
			progressbar_draw(bar, (double)j / weights->shape[1] * 0.2 + 0.8); // TODO: same as above
#endif 
		}
	}
	
#ifdef DRAW_PROGRESS
	printf("\033[u"); // Restore cursor position
	printf(" \x1B[36m"); // Color cyan 
	double elapsed_time = (current_time_millis() - bar->time_started) / 1000.0; 
	printf(bar->digit_format, elapsed_time); 
	network_operation_time += elapsed_time; 
	printf(" \x1B[35m(%.2f)", network_operation_time); // Color magenta, total time
	printf("\x1B[0m\n"); // Reset color and newline 
	progressbar_free(bar); 
#endif 
}

ND_TYPE layer_relu(ND_TYPE* value) { 
	return *value > 0 ? *value : 0; 
}

ND_TYPE layer_softmax(ND_TYPE* value) {
	return exp(value[0]) / value[1]; 
}