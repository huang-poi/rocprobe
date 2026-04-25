HIPCC ?= hipcc
all: vec_add
vec_add: main.cpp
	$(HIPCC) -O2 --offload-arch=gfx942 -o $@ $<
clean:
	rm -f vec_add
