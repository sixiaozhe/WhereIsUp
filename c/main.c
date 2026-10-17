#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "imutil.h"
#include "nn_inference.h"
#include "weights.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static const char *DIRNAME[4] = {"上", "右", "下", "左"};
static float input_buf[3 * 64 * 64];

static void build_input(const unsigned char *hwc) {
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++)
            for (int c = 0; c < 3; c++) {
                float v = hwc[(y * 64 + x) * 3 + c] / 255.0f;
                input_buf[c * 64 * 64 + y * 64 + x] = (v - NORM_MEAN[c]) / NORM_STD[c];
            }
}

static void softmax4(const float *in, float *out) {
    float m = in[0];
    for (int i = 1; i < 4; i++) if (in[i] > m) m = in[i];
    float s = 0.0f;
    for (int i = 0; i < 4; i++) { out[i] = expf(in[i] - m); s += out[i]; }
    for (int i = 0; i < 4; i++) out[i] /= s;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: whereisup <image> [-o fixed.png] [--no-tta]\n"
                        "       whereisup --ppm <file.ppm> [--logits] [--no-tta]\n");
        return 2;
    }
    int ppm_mode = (strcmp(argv[1], "--ppm") == 0);
    int logits_mode = 0;
    int tta = 1;
    const char *out_path = NULL;
    const char *in_path = NULL;

    if (ppm_mode) {
        if (argc < 3) { fprintf(stderr, "whereisup: --ppm needs a file\n"); return 2; }
        in_path = argv[2];
        for (int i = 3; i < argc; i++) {
            if (!strcmp(argv[i], "--logits")) logits_mode = 1;
            else if (!strcmp(argv[i], "--no-tta")) tta = 0;
        }
    } else {
        in_path = argv[1];
        for (int i = 2; i < argc; i++) {
            if (!strcmp(argv[i], "-o") && i + 1 < argc) out_path = argv[++i];
            else if (!strcmp(argv[i], "--no-tta")) tta = 0;
        }
    }

    int ow = 0, oh = 0, oc = 0;
    unsigned char *small = NULL;      /* 64x64x3 HWC */
    unsigned char *full = NULL;       /* 原分辨率 HWC，用于纠偏保存 */

    if (ppm_mode) {
        small = read_ppm(in_path, &ow, &oh, &oc);
        if (!small || ow != 64 || oh != 64 || oc != 3) {
            fprintf(stderr, "whereisup: %s must be a 64x64 P6 PPM\n", in_path);
            free(small);
            return 1;
        }
    } else {
        int w = 0, h = 0, c = 0;
        unsigned char *img = stbi_load(in_path, &w, &h, &c, 3);
        if (!img) { fprintf(stderr, "whereisup: cannot decode %s\n", in_path); return 1; }
        if (w < 8 || h < 8) { fprintf(stderr, "whereisup: image too small (%dx%d)\n", w, h); stbi_image_free(img); return 1; }
        full = img;
        small = malloc(64 * 64 * 3);
        bilinear_resize(img, w, h, 3, small, 64, 64);
    }

    float logits[4], probs[4];
    if (tta) {
        /* 四方向 TTA：输入旋转 r*90° 后类别会变成 (k+r)%4，反向对齐再平均。 */
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        unsigned char *rot = malloc(64 * 64 * 3);
        for (int r = 0; r < 4; r++) {
            float l[4], p[4];
            rotate90cw(small, 64, 64, 3, rot, r);
            build_input(rot);
            forward(input_buf, l, p);
            for (int j = 0; j < 4; j++) acc[(j - r + 4) % 4] += l[j];
        }
        free(rot);
        for (int j = 0; j < 4; j++) logits[j] = acc[j] / 4.0f;
        softmax4(logits, probs);
    } else {
        build_input(small);
        forward(input_buf, logits, probs);
    }

    if (logits_mode) {
        printf("%.6f %.6f %.6f %.6f\n", logits[0], logits[1], logits[2], logits[3]);
    } else {
        int k = 0;
        for (int i = 1; i < 4; i++) if (probs[i] > probs[k]) k = i;
        printf("probs: %.4f %.4f %.4f %.4f\n", probs[0], probs[1], probs[2], probs[3]);
        printf("上边在: %s (k=%d, 纠正需逆时针 %d 度)\n", DIRNAME[k], k, k * 90);
        if (!ppm_mode && out_path) {
            int w = 0, h = 0, c = 0;
            stbi_info(in_path, &w, &h, &c);
            unsigned char *fixed = malloc((size_t)w * h * 3);
            rotate90cw(full, w, h, 3, fixed, (4 - k) % 4);
            if (!stbi_write_png(out_path, (k == 0 || k == 2) ? w : h,
                                (k == 0 || k == 2) ? h : w, 3, fixed,
                                ((k == 0 || k == 2) ? w : h) * 3))
                fprintf(stderr, "whereisup: cannot write %s\n", out_path);
            else
                printf("纠正图已保存: %s\n", out_path);
            free(fixed);
        }
    }

    free(small);
    if (full) stbi_image_free(full);
    return 0;
}
