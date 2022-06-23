#include <stdio.h> 
#include "network.h"

int main(int argc, char** argv) {
	network *network = network_create("data/network.nn", "data.json");
	network_free(network); 
}