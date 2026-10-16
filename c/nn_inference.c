#include "nn_inference.h"
#include "weights.h"
#include <math.h>

static void conv3x3(const float *in, int C, int H, int W,
                    const float *K, int OC, const float *B, float *out) {
    for (int oc = 0; oc < OC; oc++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                float acc = B[oc];
                for (int c = 0; c < C; c++) {
                    for (int i = 0; i < 3; i++) {
                        int yy = y + i - 1;
                        if (yy < 0 || yy >= H) continue;
                        for (int j = 0; j < 3; j++) {
                            int xx = x + j - 1;
                            if (xx < 0 || xx >= W) continue;
                            acc += in[c * H * W + yy * W + xx] *
                                   K[((oc * C + c) * 3 + i) * 3 + j];
                        }
                    }
                }
                out[oc * H * W + y * W + x] = acc;
            }
        }
    }
}

static void relu_inplace(float *a, int n) {
    for (int i = 0; i < n; i++) if (a[i] < 0.0f) a[i] = 0.0f;
}

static void maxpool2(const float *in, int C, int H, int W, float *out) {
    int oh = H / 2, ow = W / 2;
    for (int c = 0; c < C; c++) {
        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                float m = in[c * H * W + (2 * y) * W + 2 * x];
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 2; j++) {
                        float v = in[c * H * W + (2 * y + i) * W + (2 * x + j)];
                        if (v > m) m = v;
                    }
                out[c * oh * ow + y * ow + x] = m;
            }
        }
    }
}

static void linear(const float *in, int IN, const float *Wm, int OUT,
                   const float *b, float *out) {
    for (int o = 0; o < OUT; o++) {
        float acc = b[o];
        for (int i = 0; i < IN; i++) acc += in[i] * Wm[o * IN + i];
        out[o] = acc;
    }
}

static void softmax(const float *in, int n, float *out) {
    float m = in[0];
    for (int i = 1; i < n; i++) if (in[i] > m) m = in[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { out[i] = expf(in[i] - m); s += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= s;
}

void forward(const float *x, float *logits, float *probs) {
    static float c1[8 * 64 * 64];
    static float p1[8 * 32 * 32];
    static float c2[16 * 32 * 32];
    static float p2[16 * 16 * 16];
    conv3x3(x, 3, 64, 64, W1, 8, B1, c1);
    relu_inplace(c1, 8 * 64 * 64);
    maxpool2(c1, 8, 64, 64, p1);
    conv3x3(p1, 8, 32, 32, W2, 16, B2, c2);
    relu_inplace(c2, 16 * 32 * 32);
    maxpool2(c2, 16, 32, 32, p2);
    linear(p2, 16 * 16 * 16, W3, 4, B3, logits);
    softmax(logits, 4, probs);
}
