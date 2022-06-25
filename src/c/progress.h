#ifndef PROGRESS_H
#define PROGRESS_H 

#include <time.h> 

struct progressbar {
	int width;
	int digits; 
	char *digit_format; 
	time_t time_started; 
};

struct progressbar* progressbar_create(int width, int whole_digits, int precision);
void progressbar_free(struct progressbar* bar); 
void progressbar_draw(struct progressbar* bar,  double percentage); 

time_t current_time_millis(); 

#endif 