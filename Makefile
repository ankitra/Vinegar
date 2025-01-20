src_dir = src
external = external

compile_flags = -D DEBUG -D INSTRUMENT -I $(external)/

.PHONY: clean all

all: recipe1 recipe2

clean:
	rm -f recipe1 recipe2 *.d $(src_dir)/*.d

recipe%: $(src_dir)/recipe%.cu $(src_dir)/vinegar.h
	nvcc $(compile_flags) -MD -o $@ $<

-include $(wildcard src/*.d)
