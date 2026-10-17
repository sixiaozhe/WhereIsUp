/* 纯前向基准：测量单次 forward 与四方向 TTA 的耗时与常驻内存。
 * 用法: make -C c bench && ./c/bench
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <sys/resource.h>
#include <time.h>

#include "nn_inference.h"

static float x[3 * 64 * 64];

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

int main(void) {
    for (int i = 0; i < 3 * 64 * 64; i++) x[i] = (float)((i * 37) % 255) / 255.0f - 0.5f;
    float logits[4], probs[4];

    for (int i = 0; i < 50; i++) forward(x, logits, probs); /* 预热 */

    const int N = 2000;
    double t0 = now();
    for (int i = 0; i < N; i++) forward(x, logits, probs);
    double us = (now() - t0) / N * 1e6;

    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    printf("forward : %.1f us/call\n", us);
    printf("4x TTA  : %.1f us\n", us * 4.0);
    printf("max RSS : %.2f MB\n", (double)ru.ru_maxrss / 1024.0);
    return 0;
}
