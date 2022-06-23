pre-build:
ifeq ("$(wildcard lib/json-parser/*)", "") 
	git submodule init
	git submodule update
endif

ifeq ("$(wildcard data/network.nn)", "") 
	$(MAKE) nn
endif 
	
ifeq ("$(wildcard lib/libjpeg/djpeg)", "")
	$(MAKE) libjpeg
endif

ifeq ("$(wildcard obj/*)", "") 
	mkdir -p obj 
	mkdir -p obj/c 
	mkdir -p obj/python 
	mkdir -p bin 
endif
	
nn: 
	python3 src/python/network_operations.py -save data/network.nn

libjpeg: 
	cd lib/libjpeg ; \
	./configure ; \
	$(MAKE) ; \
	$(MAKE) test ; \
	sudo make install			

c: pre-build
	gcc -o obj/c/vgg.o -c src/c/vgg.c 
	gcc -o obj/c/network.o -c src/c/network.c 
	gcc -o obj/c/layer.o -c src/c/layer.c 
	gcc -o obj/c/ndarray.o -c src/c/ndarray.c 
	gcc -o obj/c/image.o -c src/c/image.c
	gcc -o bin/vgg obj/c/vgg.o obj/c/network.o obj/c/layer.o obj/c/ndarray.o obj/c/image.o
	./bin/vgg

python: 
	python3 src/python/vgg.py

json: pre-build
	gcc -o obj/c/json.o -c lib/json-parser/json.c
	gcc -o obj/c/json_test.o -Ilib/json-parser -c src/json_test.c
	gcc -o bin/c/json_test obj/c/json_test.o obj/c/json.o -lm
	./bin/json_test data/imagenet_class_index.json

clean: 
	rm -f vgg
	rm -f json_test 
	rm -f obj/*.o
	rm -f bin/*
	rm -rf obj
	rm -rf bin 