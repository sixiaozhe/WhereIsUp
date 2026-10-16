#include "imutil.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

void bilinear_resize(const unsigned char *src, int sw, int sh, int c,
                     unsigned char *dst, int dw, int dh) {
    for (int y = 0; y < dh; y++) {
        float sy = ((y + 0.5f) * sh / dh) - 0.5f;
        int y0 = (int)floorf(sy);
        float fy = sy - y0;
        int y1 = y0 + 1;
        y0 = clampi(y0, 0, sh - 1); y1 = clampi(y1, 0, sh - 1);
        for (int x = 0; x < dw; x++) {
            float sx = ((x + 0.5f) * sw / dw) - 0.5f;
            int x0 = (int)floorf(sx);
            float fx = sx - x0;
            int x1 = x0 + 1;
            x0 = clampi(x0, 0, sw - 1); x1 = clampi(x1, 0, sw - 1);
            for (int ch = 0; ch < c; ch++) {
                float v00 = src[(y0 * sw + x0) * c + ch];
                float v01 = src[(y0 * sw + x1) * c + ch];
                float v10 = src[(y1 * sw + x0) * c + ch];
                float v11 = src[(y1 * sw + x1) * c + ch];
                float top = v00 + (v01 - v00) * fx;
                float bot = v10 + (v11 - v10) * fx;
                float v = top + (bot - top) * fy;
                dst[(y * dw + x) * c + ch] = (unsigned char)clampi((int)(v + 0.5f), 0, 255);
            }
        }
    }
}

void rotate90cw(const unsigned char *src, int w, int h, int c,
                unsigned char *dst, int k) {
    k = ((k % 4) + 4) % 4;
    if (k == 0) { memcpy(dst, src, (size_t)w * h * c); return; }
    unsigned char *cur = malloc((size_t)w * h * c);
    memcpy(cur, src, (size_t)w * h * c);
    int cw = w, chh = h;
    for (int t = 0; t < k; t++) {
        unsigned char *nxt = (t == k - 1) ? dst : malloc((size_t)chh * cw * c);
        for (int y = 0; y < chh; y++)
            for (int x = 0; x < cw; x++)
                for (int ch = 0; ch < c; ch++)
                    nxt[(x * chh + (chh - 1 - y)) * c + ch] = cur[(y * cw + x) * c + ch];
        free(cur);
        cur = nxt;
        int tmp = cw; cw = chh; chh = tmp;
    }
}

unsigned char *read_ppm(const char *path, int *w, int *h, int *c) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[3] = {0};
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) { fclose(f); return NULL; }
    int vals[3], got = 0;
    while (got < 3) {
        int ch = fgetc(f);
        if (ch == EOF) { fclose(f); return NULL; }
        if (ch == '#') { while ((ch = fgetc(f)) != EOF && ch != '\n') {} continue; }
        if (isspace(ch)) continue;
        ungetc(ch, f);
        if (fscanf(f, "%d", &vals[got]) != 1) { fclose(f); return NULL; }
        got++;
    }
    fgetc(f);
    int W = vals[0], H = vals[1], CH = vals[2];
    unsigned char *buf = malloc((size_t)W * H * CH);
    if (fread(buf, 1, (size_t)W * H * CH, f) != (size_t)W * H * CH) { free(buf); fclose(f); return NULL; }
    fclose(f);
    *w = W; *h = H; *c = CH;
    return buf;
}

int write_ppm(const char *path, const unsigned char *data, int w, int h, int c) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P6\n%d %d\n%d\n", w, h, c);
    fwrite(data, 1, (size_t)w * h * c, f);
    fclose(f);
    return 0;
}
