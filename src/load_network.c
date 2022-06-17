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
	FILE *file = fopen("/Users/bora/Desktop/VGG-Simulation/data/network.nn", "rb");
	if (file == NULL) {
		fprintf(stderr, "Error opening file!\n");
	}
	
	int magic_number = read_int(file, buffer); 
    //printf("Magic number: %d \n", magic_number);
	int number_of_layers = read_int(file, buffer);
    //printf("Number of layers: %d \n", number_of_layers);
	for (int layer_index = 0; layer_index < number_of_layers; layer_index++) {
		int layer_type = read_int(file, buffer);
        //printf("Layer index: %d \n", layer_index);
        //printf("Layer type: %d \n", layer_type);
        if (layer_type == 1) {
            int type_of_activation = read_int(file, buffer);
            //printf("Type of activation: %d \n", type_of_activation);
        }
        else if (layer_type == 4) {
            int type_of_activation = read_int(file, buffer);
            //printf("Type of activation: %d \n", type_of_activation);
        }
        int number_of_weight_indices = read_int(file, buffer);
        //printf("Number of weight indices: %d \n", number_of_weight_indices);
        for (int weight_index = 0; weight_index < number_of_weight_indices; weight_index++) {
            int length_of_weights_shape = read_int(file, buffer);
            //printf("Length of weights shape: %d \n", length_of_weights_shape);
            int* weights_shape = (int*) malloc(length_of_weights_shape*sizeof(int));
            for (int shape_index = 0; shape_index < length_of_weights_shape; shape_index++) {
                int shape_value = read_int(file, buffer);
                //printf("Shape value: %d \n", shape_value);
                weights_shape[shape_index] = shape_value;
            }
            int number_of_data_stored_in_shape = 1;
            for (int i = 0; i < length_of_weights_shape; i++) {
                number_of_data_stored_in_shape *= weights_shape[i];
            }
            for (int i = 0; i < number_of_data_stored_in_shape; i++) {
                float ba = read_float(file, buffer);
                //printf("Data: %f \n", ba);
            }
        }
	}
	free(buffer); 
}

/*
int main() {

    load_network();
}
*/