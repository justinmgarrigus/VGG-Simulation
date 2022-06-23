#include <stdio.h> 
#include <stdlib.h>
#include "network.h"
#include "image.h" 

int main(int argc, char** argv) {
	printf("Loading image\n"); 
	image *img = image_load("data/dog.jpg");
	
	printf("Creating network\n"); 
	network *network = network_create("data/network.nn", "data.json");
	
	int *length = malloc(sizeof(int) * 4); 
	length[0] = 1; length[1] = 224; length[2] = 224; length[3] = 3; 
	ndarray *input = ndarray_create(4, length); 
	free(length); 
	printf("Shape: "); 
	ndarray_fprint(input, stdout); printf("\n"); 
	
	printf("Writing image to input\n"); 
	for (int y = 0; y < 224; y++) {
		for (int x = 0; x < 224; x++) {
			int color = image_pos_color(img, x * (img->width / 224), y * (img->height / 224)); 
			ndarray_set_val_param(input, color | 0xFF, 0, x, y, 2);
			ndarray_set_val_param(input, color >> 8 | 0xFF, 0, x, y, 1);
			ndarray_set_val_param(input, color >> 16 | 0xFF, 0, x, y, 0);
		}
	}
	image_free(img); 
	
	printf("Feed forward\n"); 
	network_feedforward(network, input); 
	network_free(network); 
}