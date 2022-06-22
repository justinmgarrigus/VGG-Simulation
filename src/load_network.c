#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int create_int(unsigned char* buffer) {
	return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];  
}

float create_float(unsigned char* buffer) {
    float result;
    memcpy(&result, buffer, sizeof(float));
	return result;
}

int read_int(FILE* file, unsigned char* buffer) {
	int length = sizeof(unsigned char) * 4; 
	fread(buffer, length, 1, file);
	return create_int(buffer); 
}

float read_float(FILE* file, unsigned char* buffer) {
	int length = sizeof(unsigned char) * 4; 
	fread(buffer, length, 1, file);
	return create_float(buffer); 
}

void load_network() {
	int length = sizeof(unsigned char) * 4;
	unsigned char *buffer = malloc(length);
    int *magic_number_ptr = malloc(sizeof(int));
    int *num_layers_ptr = malloc(sizeof(int));
    int *len_of_weights_shape_ptr;
    int* weights_shape_ptr;
    int *shape_value_ptr;
    float *data_ptr;
	FILE *file = fopen("/Users/bora/Desktop/VGG-Simulation/data/network.nn", "rb");
	if (file == NULL) {
		fprintf(stderr, "Error opening file!\n");
	}
	
	int magic_number = read_int(file, buffer);
    *magic_number_ptr = magic_number;
    printf("Magic number: %d \n", *magic_number_ptr);
	int number_of_layers = read_int(file, buffer);
    *num_layers_ptr = number_of_layers;
    printf("Number of layers: %d \n", *num_layers_ptr);
    int *layer_type_ptr = malloc(sizeof(int) * (*num_layers_ptr));
    int *num_weight_indices_ptr = malloc(sizeof(int) * (*num_layers_ptr));
    int *activation_type_ptr = malloc(sizeof(int) * (*num_layers_ptr));
	for (int layer_index = 0; layer_index < number_of_layers; layer_index++) {
		int layer_type = read_int(file, buffer);
        layer_type_ptr[layer_index] = layer_type;
        printf("Layer index: %d \n", layer_index);
        printf("Layer type: %d \n", layer_type_ptr[layer_index]);
        if (layer_type == 1) {
            int type_of_activation = read_int(file, buffer);
            activation_type_ptr[layer_index] = type_of_activation;
            printf("Type of activation: %d \n", activation_type_ptr[layer_index]);
        }
        else if (layer_type == 4) {
            int type_of_activation = read_int(file, buffer);
            activation_type_ptr[layer_index] = type_of_activation;
            printf("Type of activation: %d \n", activation_type_ptr[layer_index]);
        }
        int number_of_weight_indices = read_int(file, buffer);
        num_weight_indices_ptr[layer_index] = number_of_weight_indices;
        printf("Number of weight indices: %d \n", (num_weight_indices_ptr[layer_index]));
        len_of_weights_shape_ptr = malloc(sizeof(int) * number_of_weight_indices);
        for (int weight_index = 0; weight_index < number_of_weight_indices; weight_index++) {
            int length_of_weights_shape = read_int(file, buffer);
            len_of_weights_shape_ptr[weight_index] = length_of_weights_shape;
            printf("Length of weights shape: %d \n", len_of_weights_shape_ptr[weight_index]);
            weights_shape_ptr = (int*) malloc(length_of_weights_shape*sizeof(int));
            for (int shape_index = 0; shape_index < length_of_weights_shape; shape_index++) {
                int shape_value = read_int(file, buffer);
                weights_shape_ptr[shape_index] = shape_value;
                printf("Shape value: %d \n", weights_shape_ptr[shape_index]);
            }
            int number_of_data_stored_in_shape = 1;
            for (int i = 0; i < length_of_weights_shape; i++) {
                number_of_data_stored_in_shape *= weights_shape_ptr[i];
            }
            data_ptr = (float*) malloc(number_of_data_stored_in_shape*sizeof(float));
            for (int i = 0; i < number_of_data_stored_in_shape; i++) {
                float ba = read_float(file, buffer);
                data_ptr[i] = ba;
                printf("Data: %f \n", data_ptr[i]);
            }
        }
	}
	free(buffer);
    free(magic_number_ptr);
    free(num_layers_ptr);
    free(layer_type_ptr);
    free(num_weight_indices_ptr);
    free(len_of_weights_shape_ptr);
    free(weights_shape_ptr);
    free(data_ptr);
    free(activation_type_ptr);
    fclose(file);
}

/*
int main() {

    load_network();
}
*/