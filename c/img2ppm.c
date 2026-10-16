#include <stdio.h>
#include <stdlib.h>
#include "imutil.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: img2ppm <in> <out.ppm> [size=64]\n"); return 2; }
    int size = (argc >= 4) ? atoi(argv[3]) : 64;
    int w, h, c;
    unsigned char *img = stbi_load(argv[1], &w, &h, &c, 3);
    if (!img) { fprintf(stderr, "img2ppm: cannot decode %s\n", argv[1]); return 1; }
    unsigned char *dst = malloc((size_t)size * size * 3);
    bilinear_resize(img, w, h, 3, dst, size, size);
    stbi_image_free(img);
    int rc = write_ppm(argv[2], dst, size, size, 3);
    free(dst);
    if (rc) { fprintf(stderr, "img2ppm: cannot write %s\n", argv[2]); return 1; }
    return 0;
}
