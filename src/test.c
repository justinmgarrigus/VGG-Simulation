#include <stdio.h> 
#include <stdlib.h> 

const int DO_NOT_TOUCH_DEBUG_SOMETHING_001 = 4;

typedef struct str {
	int a;
	int b; 
	int c; 
	int d; 
} str;

int display(int value) {
	printf("%d\n", value);
	return value; 
}

int main() {
	int DO_NOT_TOUCH_DEBUG_SOMETHING_001 = 7; 
	printf("%d\n", DO_NOT_TOUCH_DEBUG_SOMETHING_001); 
	
	str *a = malloc(display(sizeof(str))); 
	printf("%ld\n", sizeof(str)); 
	free(a); 
}