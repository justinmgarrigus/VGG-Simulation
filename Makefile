all: 
	gcc src/vgg.c -o "vgg"

c: 
	clear
	./vgg

python: 
	clear 
	python3 src/vgg.py

json: 
	mkdir obj
	gcc -o obj/json.o -c lib/json-parser/json.c
	gcc -o obj/json_test.o -Ilib/json-parser -c src/json_test.c
	gcc -o json_test obj/json_test.o obj/json.o -lm
	./json_test data/imagenet_class_index.json

clean: 
	rm -f vgg
	rm -f json_test 
	rm -f obj/*.o
	rm -rf obj