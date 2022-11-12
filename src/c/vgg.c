#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <math.h> 
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h> 
#include <dirent.h> 
#include "network.h"
#include "image.h" 

void vgg_image(char* file_name, ndarray* input) {
	printf("Loading image: %s\n", file_name); 
	image *img = image_load(file_name);
	
	int sum_red = 0; 
	int sum_green = 0; 
	int sum_blue = 0; 
	int size = img->width * img->height; 
	for (int i = 0; i < size; i++) {
		int color = img->colors[i]; 
		sum_red   += color >> 16 & 0xFF; 
		sum_green += color >> 8  & 0xFF; 
		sum_blue  += color       & 0xFF; 
	}
	int avg_red = sum_red / size; 
	int avg_green = sum_green / size; 
	int avg_blue = sum_blue / size;
	
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
}

void alexnet_image(char* file_name, ndarray* input) {
	printf("Loading image: %s\n", file_name); 
	image *img = image_load(file_name); 
	int size = img->width * img->height * 3; 
	
	// Average of all colors in image 
	int elem_sum = 0;
	for (int i = 0; i < img->width * img->height; i++) {
		int color = img->colors[i]; 
		int red   = color >> 16 & 0xFF, 
			green = color >> 8  & 0xFF, 
			blue  = color       & 0xFF;
		elem_sum += red + green + blue;
	}
	float mean = (float)elem_sum / size; 
	
	// Standard deviation 
	float deviation_sum = 0; 
	for (int i = 0; i < img->width * img->height; i++) {
		int color = img->colors[i]; 
		int red   = color >> 16 & 0xFF, 
			green = color >> 8  & 0xFF, 
			blue  = color       & 0xFF;
		deviation_sum += powf(red   - mean, 2); 
		deviation_sum += powf(green - mean, 2); 
		deviation_sum += powf(blue  - mean, 2);
	}
	float dev = sqrtf(deviation_sum / size); 
	float sqrt_size = 1.0f / sqrtf(size); 
	float adjusted_dev = (dev > sqrt_size) ? dev : sqrt_size; 
	
	for (int x = 0; x < input->shape[1]; x++) {
		for (int y = 0; y < input->shape[2]; y++) {
			int image_x = (float)x / input->shape[1] * img->width;
			int image_y = (float)y / input->shape[2] * img->height;
			
			int color = image_pos_color(img, image_x, image_y);
			float red =   ((color >> 16 & 0xFF) - mean) / adjusted_dev; 
			float green = ((color >> 8  & 0xFF) - mean) / adjusted_dev; 
			float blue =  ((color       & 0xFF) - mean) / adjusted_dev;
			
			ndarray_set_val_param(input, red,   0, y, x, 0);
			ndarray_set_val_param(input, green, 0, y, x, 1);
			ndarray_set_val_param(input, blue,  0, y, x, 2);
		}
	}
	image_free(img); 
	
	printf("Mean: %f, dev: %f, entropy: %f\n", mean, dev, ndarray_entropy(input)); 
}

int main(int argc, char** argv) {
	if (argc < 2 || argc > 3) {
		printf("Format: ./vgg <model> [<image.jpg>]\n"); 
		printf("  Model types: -vgg16 -alexnet\n");  
		exit(1); 
	}
	
	if (argv[1][0] == '-') 
		argv[1] = argv[1] + 1;
	
	enum model_type type; 
	if (strcmp(argv[1], "alexnet") == 0)
		type = model_alexnet;
	else if (strcmp(argv[1], "vgg16") == 0)
		type = model_vgg16;
	else {
		printf("Unrecognized model type specified: '%s'\n", argv[1]);
		exit(1); 
	}
	
	int device_count = 0; 
	int device_error = cudaGetDeviceCount(&device_count); 
	printf("Device count: %d\n", device_count); 
	if (device_count == 0 || device_error != cudaSuccess) {
		printf("No devices are connected: %s(%d)\n", cudaGetErrorName(device_error), device_error); 
		exit(1); 
	}
	
	network *network = network_create(type);
	void (*preprocess)(char* file_name, ndarray* input);
	ndarray *input; 
	if (type == model_alexnet) {
		preprocess = alexnet_image; 
		int length[4] = { 1, 227, 227, 3 }; 
		input = ndarray_create(4, length); 
	}
	else {
		preprocess = vgg_image;
		int length[4] = { 1, 224, 224, 3 };
		input = ndarray_create(4, length); 
	}
	
#if defined(SAVE_INTERMEDIATE) 
	conv_layer_counter = 1;  
#endif 
	
	if (argc == 3) {
		char *path = argv[2]; 
		struct stat path_stat; 
		stat(path, &path_stat); 
		
		if (S_ISREG(path_stat.st_mode)) {
			// The argument is just a regular file, not a directory.
			preprocess(argv[2], input); 
			network_feedforward(network, input); printf("\n");
			network_decode_output(network);
		}
		else {
			// The argument is a directory containing image files. 
			DIR *d;
			struct dirent *dir;
			d = opendir(argv[2]); 
			char buffer[256]; 
			if (d) {
				while ((dir = readdir(d)) != NULL) { 
					if (dir->d_name[0] == '.')
						continue; 
					
					conv_layer_counter = 1; 
					sprintf(buffer, "%s/%s", argv[2], dir->d_name); 
					preprocess(buffer, input); 
					network_feedforward(network, input); printf("\n");
					network_decode_output(network);
					printf("\n\n"); 
				}
				closedir(d); 
			}
			else {
				printf("Error opening directory: '%s'\n", argv[2]);
			}
		}
	}
	else {
		printf("\n\n"); 
		while (1) {
			char file_name[256]; 
			printf("Enter a file name or 'quit': dog.jpg\n"); 
			// scanf("%s", file_name); 
			sprintf(file_name, "dog.jpg"); 
			
			if (strcmp(file_name, "quit") == 0) 
				break; 
			
			char buffer[256]; 
			strcpy(buffer, "data/"); 
			strcat(buffer, file_name); 
			
#if defined(_MSC_VER)
			strcat(buffer, ".ppm"); 
#endif
			
			preprocess(buffer, input); 
			network_feedforward(network, input); printf("\n"); 
			network_decode_output(network); printf("\n\n"); 
			
			break; 
		}
	}
	
	network_free(network); 
}