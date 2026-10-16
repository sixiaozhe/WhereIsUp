CC ?= cc
CFLAGS ?= -std=c99 -O3 -Wall -Wextra
LDLIBS ?= -lm

TOOLS = img2ppm whereisup

all: $(TOOLS)

whereisup: main.c nn_inference.c imutil.c weights.h nn_inference.h imutil.h
	$(CC) $(CFLAGS) -o $@ main.c nn_inference.c imutil.c $(LDLIBS)

img2ppm: img2ppm.c imutil.c imutil.h
	$(CC) $(CFLAGS) -o $@ img2ppm.c imutil.c $(LDLIBS)

clean:
	rm -f $(TOOLS)
