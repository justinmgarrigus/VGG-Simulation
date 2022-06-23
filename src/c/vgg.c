#include <stdio.h> 
#include <stdlib.h>
#include "network.h"
#include "image.h" 

int main(int argc, char** argv) {
	image *img = image_load("data/dog.jpg"); 
	image_free(img); 
	
//	network *network = network_create("data/network.nn", "data.json");
	
//	int *length = malloc(sizeof(int) * 3); 
//	length[0] = 224; length[1] = 224; length[2] = 3; 
//	ndarray *input = ndarray_create(3, length); 
//	free(length); 
//	
//	network_feedforward(network, input); 
//	network_free(network); 
}