#ifndef IMUTIL_H
#define IMUTIL_H

/* 双线性缩放: src/dst 均为 HWC, 通道数 c。dst 尺寸 dw x dh。 */
void bilinear_resize(const unsigned char *src, int sw, int sh, int c,
                     unsigned char *dst, int dw, int dh);

/* 顺时针旋转 k*90 度。dst 需调用者分配 w*h*c 字节。 */
void rotate90cw(const unsigned char *src, int w, int h, int c,
                unsigned char *dst, int k);

/* 读取二进制 P6 的 PPM。返回 malloc 的缓冲区, 失败返回 NULL。 */
unsigned char *read_ppm(const char *path, int *w, int *h, int *c);

/* 写二进制 P6 的 PPM。成功返回 0。 */
int write_ppm(const char *path, const unsigned char *data, int w, int h, int c);

#endif
