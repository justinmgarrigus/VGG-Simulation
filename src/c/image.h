#ifndef IMAGE_H 
#define IMAGE_H 

typedef struct image {
	int width, height; 
	int *colors;
} image; 

struct image_iterator {
	int x, y; 
	int width_max, height_max;
	float width_stride, height_stride; 
};

image* image_load(char* file_name); 
void image_free(image* img); 
struct image_iterator* image_create_iter(image* img, int width, int height);
int image_iter_color(image* img, struct image_iterator* iter);
int image_pos_color(image* img, int x, int y); 

#endif 