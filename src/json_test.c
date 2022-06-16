#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

char** get_labels(json_value* json, int* length_output) {
	int length = json->u.object.length;
	char **labels = malloc(sizeof(char*) * length); 
	for (int index = 0; index < length; index++) {
		// TODO this function is unsafe; check the data before retrieving it to ensure 
		// it is in the correct format! 
		json_object_entry *entry = &(json->u.object.values[index]); 
		labels[index] = entry->value->u.array.values[1]->u.string.ptr; 
	}
	
	*length_output = length; 
	return labels; 
}

int main() {
	char *file_name = "data/imagenet_class_index.json"; 
	
	struct stat file_status; 
	if (stat(file_name, &file_status) != 0) {
		fprintf(stderr, "File %s is not found\n", file_name); 
		return 1; 
	}
	
	int file_size = file_status.st_size; 
	char *file_contents = malloc(sizeof(char) * file_size);
	
	FILE *fp = fopen(file_name, "rt");
	if (fp == NULL) {
		fprintf(stderr, "Unable to open %s\n", file_name); 
		free(file_contents); 
		return 1; 
	}
	
	if (fread (file_contents, file_size, 1, fp) != 1) {
		fprintf(stderr, "Unable to read contents of %s\n", file_name); 
		fclose(fp); 
		free(file_contents); 
		return 1; 
	}
	fclose(fp); 
	
	json_value *json = json_parse((json_char*)file_contents, file_size); 
	if (json == NULL) {
		fprintf(stderr, "Unable to parse data\n"); 
		free(file_contents); 
		return 1; 
	}
	
	int length; 
	char **labels = get_labels(json, &length); 
	printf("Element 300 has value %s; total length of %d\n", labels[300], length);
	
	json_value_free(json);
	free(file_contents); 
	free(labels); 
}