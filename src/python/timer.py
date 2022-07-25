import time 

elapsed_time = 0 
start_time = 0 

def start(): 
	global start_time
	start_time = time.time() 
	
def stop():
	global elapsed_time 
	elapsed_time += time.time() - start_time 

def elapsed():
	return round(elapsed_time, 2) 