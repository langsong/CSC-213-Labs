CC := nvcc
CFLAGS := -g $(shell sdl2-config --cflags) -ccbin clang-3.8 -arch sm_20
LDFLAGS := $(shell sdl2-config --libs) -lm

all: galaxy

clean:
	rm -f galaxy

galaxy: main.cu gui.cu gui.h
	$(CC) $(CFLAGS) -o galaxy main.cu gui.cu $(LDFLAGS)
