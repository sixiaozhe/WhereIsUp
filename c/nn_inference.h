#ifndef NN_INFERENCE_H
#define NN_INFERENCE_H

#define IMG_C 3
#define IMG_H 64
#define IMG_W 64
#define N_CLASSES 4

/* 输入 x 为 CHW 归一化后的 float[3*64*64]。
 * logits[4] 原始分数; probs[4] softmax 概率。 */
void forward(const float *x, float *logits, float *probs);

#endif
