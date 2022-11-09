FLAGS = -DTENSORCORE -DSAVE_INTERMEDIATE
#FLAGS = -DTENSORCORE
INCLUDES = -Ilib/Common -I/usr/local/cuda/include -Isrc/c -Isrc/gpu -lcudadevrt -lcudart_static -lrt -lpthread -ldl  -L"/usr/local/cuda-9.1/lib64/stubs" -L"/usr/local/cuda-9.1/lib64" 
MODEL = -vgg16 

.PHONY: internal-target external-target

pre-build:
ifeq ("$(wildcard lib/json-parser/*)", "") 
	git submodule init
	git submodule update
endif
	
ifeq ("$(wildcard lib/libjpeg/djpeg)", "")
	$(MAKE) libjpeg
endif

	mkdir -p obj 
	mkdir -p obj/c 
	mkdir -p obj/gpu
	mkdir -p obj/python 
	mkdir -p bin 
	mkdir -p conv_data

libjpeg: 
	cd lib/libjpeg ; \
	./configure ; \
	$(MAKE) ; \
	$(MAKE) test ; \
	sudo make install
	
c-compile: pre-build
	gcc -o obj/c/vgg.o -c src/c/vgg.c $(FLAGS) $(INCLUDES)
	gcc -o obj/c/network.o -c src/c/network.c -Ilib/json-parser $(FLAGS) $(INCLUDES)
	gcc -o obj/c/layer.o -c src/c/layer.c $(FLAGS) $(INCLUDES)
	nvcc -o obj/gpu/layer_gpu.o -c src/gpu/layer_gpu.cu $(FLAGS) $(INCLUDES)
	gcc -o obj/c/ndarray.o -c src/c/ndarray.c $(FLAGS) $(INCLUDES)
	gcc -o obj/c/image.o -c src/c/image.c $(FLAGS) $(INCLUDES)
	gcc -o obj/c/json.o -c lib/json-parser/json.c $(FLAGS) $(INCLUDES)
	nvcc -o obj/gpu/cudaTensorCoreGemm.o -c src/gpu/cudaTensorCoreGemm.cu $(FLAGS) $(INCLUDES) -arch=sm_70
	nvcc -o bin/vgg obj/c/vgg.o obj/c/network.o obj/c/layer.o obj/gpu/layer_gpu.o obj/c/ndarray.o obj/c/image.o obj/c/json.o obj/gpu/cudaTensorCoreGemm.o -lm --cudart shared $(FLAGS)

gemm-compile: pre-build 
	nvcc -o obj/gpu/cudaTensorCoreGemm.o -c src/gpu/cudaTensorCoreGemm.cu $(FLAGS) $(INCLUDES) -arch=sm_70
	nvcc -o obj/gpu/gemm.o -c src/gpu/gemm.cu $(FLAGS) $(INCLUDES) -arch=sm_70
	gcc -o obj/c/ndarray.o -c src/c/ndarray.c $(FLAGS) $(INCLUDES)
	nvcc -o bin/gemm obj/gpu/gemm.o obj/gpu/cudaTensorCoreGemm.o obj/c/ndarray.o --cudart shared

alexnet: c-compile
	bash -c "trap 'trap - SIGINT SIGTERM ERR; $(MAKE) c-clean; exit 1' SIGINT SIGTERM ERR; $(MAKE) c-run MODEL=alexnet"
	$(MAKE) c-clean
	
vgg16: c-compile
	bash -c "trap 'trap - SIGINT SIGTERM ERR; $(MAKE) c-clean; exit 1' SIGINT SIGTERM ERR; $(MAKE) c-run MODEL=vgg16"
	$(MAKE) c-clean

c-run:
	./bin/vgg $(MODEL)

c-clean: 
	rm -f checkpoint_files/*
	rm -rf checkpoint_files 
	rm -f _app_cuda_version_*
	rm -f _cuobjdump_list_ptx_*
	rm -f vgg.1.sm_*
	rm -f *.ptx
	rm -f *.ptxas
	rm -f gpgpu_inst_stats.txt

python: 
	python3 src/python/network.py

json: pre-build
	gcc -o obj/c/json.o -c lib/json-parser/json.c
	gcc -o obj/c/json_test.o -Ilib/json-parser -c src/c/json_test.c
	gcc -o bin/json_test obj/c/json_test.o obj/c/json.o -lm
	./bin/json_test data/imagenet_class_index.json

clean: c-clean 
	rm -rf obj 
	rm -rf bin
	rm -rf _app_cuda_version_*
	rm -rf _cuobjdump_list_ptx_*
	rm -rf *.ptx
	rm -rf *.ptxas	
