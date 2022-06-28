#include <time.h> 
#include <math.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include "progress.h" 

void progressbar_create_format_string(struct progressbar* bar, int whole_digits, int precision) {
	bar->digits = whole_digits + precision + 1;
	bar->digit_format = malloc(sizeof(char) * (bar->digits));
	sprintf(bar->digit_format, "%%0%d.%df", bar->digits, precision);
}

struct progressbar* progressbar_create(int width, int whole_digits, int precision) {
	struct progressbar* bar = malloc(sizeof(struct progressbar)); 
	bar->width = width; 
	progressbar_create_format_string(bar, whole_digits, precision); 
	
	printf("["); 
	for (int i = 0; i < width - 2; i++) 
		printf(" "); 
	printf("] "); 
	printf(bar->digit_format, 0.0);
	fflush(stdout); 
	
	bar->time_started = current_time_millis();
	return bar; 
}

void progressbar_free(struct progressbar* bar) {
	free(bar->digit_format); 
	free(bar); 
}

void progressbar_draw(struct progressbar* bar, double percentage) {
	progressbar_clear(bar); 
	printf("["); 
	
	int drawn = (bar->width - 2) * percentage;
	for (int i = 0; i < drawn; i++) 
		printf("X"); 
	for (int i = drawn; i < bar->width - 2; i++) 
		printf(" "); 
	printf("] "); 
	printf(bar->digit_format, (current_time_millis() - bar->time_started) / 1000.0);
	fflush(stdout); 
}

void progressbar_clear(struct progressbar* bar) {
	for (int i = 0; i < bar->width + bar->digits + 1; i++)
		printf("\b"); 
}

time_t current_time_millis() {
#if defined(__linux__)
	struct timespec spec; 
	clock_gettime(CLOCK_REALTIME, &spec); 
	
	time_t s = spec.tv_sec; 
	long ms = round(spec.tv_nsec / 1.0e6); 
	
	return s * 1000 + ms;
#elif defined(__WINDOWS__)
	return 0; 
#endif 
}