CC = g++  
CFLAGS = -Wall -std=c++11 

LIBRARY_DIR = /path/to/your/triton/lib

LIBRARIES := $(wildcard $(LIBRARY_DIR)/*.a $(LIBRARY_DIR)/*.so)

myprogram: client.cpp
	$(CC) $(CFLAGS) -o myprogram client.cpp -L$(LIBRARY_DIR) $(LIBRARIES) -lcurl -lz -lpthread
