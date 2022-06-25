#include <stdio.h> 
#include <stdlib.h>
#include "network.h"
#include "image.h" 

int main(int argc, char** argv) {
	network *network = network_create("data/network.nn", "data/imagenet_class_index.json");
	image *img = image_load("data/dog.jpg");
	
	int sum_red = 0; 
	int sum_green = 0; 
	int sum_blue = 0; 
	int size = img->width * img->height; 
	for (int i = 0; i < size; i++) {
		int color = img->colors[i]; 
		sum_red += color >> 16 & 0xFF; 
		sum_green += color >> 8 & 0xFF; 
		sum_blue += color & 0xFF; 
	}
	int avg_red = sum_red / size; 
	int avg_green = sum_green / size; 
	int avg_blue = sum_blue / size;
	
	int *length = malloc(sizeof(int) * 4); 
	length[0] = 1; length[1] = 224; length[2] = 224; length[3] = 3; 
	ndarray *input = ndarray_create(4, length); 
	free(length); 
	
	int num_printed = 0; 
	for (int x = 0; x < 224; x++) {
		for (int y = 0; y < 224; y++) {
			int image_x = x / 224.0f * img->width;
			int image_y = y / 224.0f * img->height;
			
			int color = image_pos_color(img, image_x, image_y);
			int red =   (color >> 16 & 0xFF) - avg_red; 
			int green = (color >> 8  & 0xFF) - avg_green; 
			int blue =  (color       & 0xFF) - avg_blue;
			
			ndarray_set_val_param(input, blue,  0, y, x, 0);
			ndarray_set_val_param(input, green, 0, y, x, 1);
			ndarray_set_val_param(input, red,   0, y, x, 2);
		}
	}
	image_free(img);
	
	network_feedforward(network, input);
	network_decode_output(network); 
	
	network_free(network); 
}