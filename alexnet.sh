#!/bin/bash 

# Display the different files we can choose from and return a status indicating
# our ability to do a complete performance test 
if [[ $# -eq 0 ]] 
then 
	files=$(find alexnet-conv-data -regex '.*[bwx]\.bin' | sort --version-sort) 
	numbers=(0 0 0 0 0) # 5 elements (1 for each Conv2D)
	for file in ${files}
	do 
		number=$(echo ${file} | tr -dc '0-9')
		index=$((${number}-1)) 
		
		# Increments value at index by 1
		numbers[${index}]=$((${numbers[${index}]}+1))
		
		echo [${number}]: ${file} 
	done 
	
	total=0
	for index in {0..4}
	do 
		subtotal=${numbers[${index}]} 
		
		if [[ ${subtotal} -ne 3 ]] 
		then 
			# A b, w, and/or x file is missing for this index. 
			echo Missing: $((${index}+1))  
		else 
			# All three bin files exist 
			echo Valid: $((${index}+1)) 
			((total+=1)) 
		fi 
	done 
	
	if [[ total -eq 5 ]] 
	then 
		# All bin files are present
		exit 0 
	else 
		# We're missing some or all bin files 
		exit 1 
	fi 

elif [[ $# -eq 1 ]]  
then 
	# If the first argument is numeric, then run only that layer
	if [[ $1 =~ ^[0-9]+$ ]]
	then 
		# Performs a single GEMM operation
		# Input: number representing the bin index we should be operating on
		function gemm {
			if [[ $# -ne 1 ]]
			then 
				echo Something went wrong 
			else
				input_file=$(find alexnet-conv-data -regex ".*[^0-9]$1_x\.bin") 
				weight_file=$(find alexnet-conv-data -regex ".*[^0-9]$1_w\.bin") 
				bias_file=$(find alexnet-conv-data -regex ".*[^0-9]$1_b\.bin")
				
				./bin/gemm ${input_file} ${weight_file} ${bias_file}
			fi 
		}
	
		if [[ $1 -eq 0 ]] 
		then 
			# In this case, we want to run the each gemm in parallel.
			./$0 > /dev/null 2>&1 # Runs ourself to ensure we have all bins 
			if [[ $? -eq 0 ]] 
			then
				for index in {1..5}
				do  
					gemm ${index} &
				done 
				
				wait
			else 
				echo Missing bin files. Run "./alexnet.sh <image.jpg>" to generate more
			fi 
		else 
			# Get the arguments that correspond to the argument 
			gemm $1 
		fi
	
	# Else if it is a jpg or ppm, process it as an input image 
	elif [[ $1 =~ .+\.(jpg|ppm)$ || -d $1 ]]
	then
		./bin/vgg -alexnet $1
	else 
		echo Error: unknown arguments 
	fi 
else
	echo Error: unknown arguments 
fi 