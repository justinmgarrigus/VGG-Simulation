pre-build: nn 
	mkdir -p obj 
	mkdir -p bin 
	
nn: 
ifeq ("$(wildcard data/network.nn)", "") 
	python3 src/network_operations.py -save network.nn
endif 

c: pre-build nn
	gcc -o obj/vgg.o -c src/vgg.c 
	gcc -o obj/network.o -c src/network.c 
	gcc -o obj/layer.o -c src/layer.c 
	gcc -o obj/ndarray.o -c src/ndarray.c 
	gcc -o bin/vgg obj/vgg.o obj/network.o obj/layer.o obj/ndarray.o
	./bin/vgg

python: 
	python3 src/vgg.py

json: pre-build
	gcc -o obj/json.o -c lib/json-parser/json.c
	gcc -o obj/json_test.o -Ilib/json-parser -c src/json_test.c
	gcc -o bin/json_test obj/json_test.o obj/json.o -lm
	./bin/json_test data/imagenet_class_index.json

clean: 
	rm -f vgg
	rm -f json_test 
	rm -f obj/*.o
	rm -f bin/*
	rm -rf obj
	rm -rf bin 
	
