#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <stddef.h> // ptrdiff_t 
#include <sys/types.h> // pid_t
#include <sys/wait.h> // wait
#include <unistd.h> // fork 
#include "image.h"

// Returns 1 if whitespace, 0 otherwise. 
int char_is_whitespace(char ch) {
	if (ch == ' ' || // space 
		ch == 9   || // tab 
		ch == 10  || // LF (line feed, newline) 
		ch == 13)	 // CR (carriage return) 
		return 1; 
	return 0; 
}

// Reads a whitespace-separated 'word' from the file
void file_read_word(char* buffer, FILE* file) {
	int index = -1; 
	char ch; 
	do {
		index++; 
		ch = fgetc(file); 
		buffer[index] = ch; 
	}
	while (ch != EOF && !char_is_whitespace(ch));
	buffer[index] = '\0'; 
}

image* image_load(char* full_file_name) {
	// Check if a ppm file already exists in this format in the data directory. 
	ptrdiff_t full_len = strlen(full_file_name);
	char *dot = strrchr(full_file_name, '.');
	ptrdiff_t name_len = dot - full_file_name; 
	char *name = malloc(name_len + 1);
	strncpy(name, full_file_name, name_len); 
	name[name_len] = '\0'; 
	ptrdiff_t exten_len = full_len - name_len; 
	char *exten = malloc(exten_len);
	strncpy(exten, dot + 1, exten_len - 1); 
	exten[exten_len] = '\0';  
	
	image *img = malloc(sizeof(image)); 
	if (strcmp(exten, "jpg") == 0) {
		char *base = "./lib/libjpeg/djpeg "; // includes space
		int command_length = strlen(base) + strlen(full_file_name); 
		char *command = malloc(command_length + 1); 
		strcpy(command, base); 
		strcat(command, full_file_name);
		command[command_length] = '\0';	
		FILE *pipe = popen(command, "r"); 
		free(command); 
		
		if (pipe == NULL) {
			fprintf(stderr, "Pipe system call failed\n"); 
			exit(1); 
		} 
		
		if (fgetc(pipe) != 'P' || fgetc(pipe) != '6') {
			fprintf(stderr, "Magic number not recognized in djpeg command!\n"); 
			exit(1); 
		}
		fgetc(pipe); // whitespace 
		
		char buffer[32];
		file_read_word(buffer, pipe); 
		img->width = atoi(buffer); 
		file_read_word(buffer, pipe); 
		img->height = atoi(buffer); 
		file_read_word(buffer, pipe);
		int dim = img->width * img->height; 		
		img->colors = malloc(sizeof(int) * dim); 
		
		int max_color = atoi(buffer);
		for (int i = 0; i < dim; i++) {
			unsigned char r = fgetc(pipe);
			unsigned char g = fgetc(pipe); 
			unsigned char b = fgetc(pipe); 
			int color = r << 16 | g << 8 | b; 
			img->colors[i] = color;
		}
		pclose(pipe);
	}
	else if (strcmp(exten, "ppm") == 0) {
		printf("It's a ppm!\n"); 
	}
	else {
		printf("Unknown type!\n"); 
	}
	
	free(name); 
	free(exten); 
	return img; 
}

void image_free(image* img) {
	free(img->colors); 
	free(img); 
}

struct image_iterator* image_create_iter(image* img, int width, int height) {
	struct image_iterator *iter = malloc(sizeof(struct image_iterator)); 
	iter->x = 0; 
	iter->y = 0; 
	iter->width_max = width; 
	iter->height_max = height; 
	iter->width_stride = img->width / width; 
	iter->height_stride = img->height / height;
	return iter; 
}

int image_iter_color(image* img, struct image_iterator* iter) {
	int color = img->colors[(int)(iter->y * iter->height_stride * iter->width_max + iter->x * iter->width_stride)]; 
	iter->x++; 
	if (iter->x == iter->width_max) {
		iter->x = 0; 
		iter->y++;
		return color; 
	}
}

int image_pos_color(image* img, int x, int y) {
	return img->colors[y * img->width + x];
}