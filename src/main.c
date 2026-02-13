#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "model_dump.h"
#include "test_vec.h"


// inputs: x, x_scale, x_zero_point, weight, weight_scale, weight_zero_point
//         bias, output_scale, output_zero_point

void linear(int32_t *x, int32_t *w, int32_t *b, int32_t *out,
            int32_t acc_scale, int32_t shift, int32_t output_zero_point,
            int input_dim, int output_dim) {
  for (int i = 0; i < output_dim; i++) {
    int64_t acc = b[i];
    for (int j = 0; j < input_dim; j++) {
      acc += x[j] * w[i*input_dim + j];
    }
    acc *= acc_scale;
    acc += (1 << (shift - 1));  // rounding
    acc >>= shift;
    // quantize to uint8
    acc = acc < -output_zero_point ? -output_zero_point : acc;
    // acc = acc < 0 ? 0 : acc;  // ReLu
    acc = acc > 255-output_zero_point ? 255-output_zero_point : acc;
    out[i] = acc;
  }
}

void relu(int32_t *x, int dim) {
  for (int i = 0; i < dim; i++) {
    x[i] = x[i] < 0 ? 0 : x[i];
  }
}

// conv2d(x, conv1_weights, conv1_bias, buf1,
//          conv1_acc_scale, conv1_shift, conv1_output_zero_point,
//          52, 12, 1, 32, 3, 3, 1, 1, 0);

void conv2d(int32_t *x, int32_t *w, int32_t *b, int32_t *out,
            int32_t acc_scale, int32_t shift, int32_t output_zero_point,
            int in_h, int in_w,
            int in_channels, int out_channels,
            int kernel_h, int kernel_w,
            int stride, int padding, char ds) {
  int output_h = 1 + (in_h - kernel_h + 2 * padding) / stride;
  int output_w = 1 + (in_w - kernel_w + 2 * padding) / stride;

  int output_idx = 0;

  for (int out_ch = 0; out_ch < out_channels; ++out_ch) {
    for (int out_r = 0; out_r < output_h; ++out_r) {
      for (int out_c = 0; out_c < output_w; ++out_c) {
        // output_idx = out_ch * output_h * output_w + out_r * output_w + out_c;
        // out[output_idx] = 0;
        int64_t acc = b[out_ch];

        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
          if (ds) in_ch = out_ch;
          for (int ker_r = 0; ker_r < kernel_h; ++ker_r) {
            for (int ker_c = 0; ker_c < kernel_w; ++ker_c) {
              int in_r = out_r * stride + ker_r - padding;
              int in_c = out_c * stride + ker_c - padding;
              int weight_idx;
              
              // padding zeros as zero point
              if (in_r >= 0 && in_r < in_h && in_c >= 0 && in_c < in_w) {
                int input_idx = in_ch * in_h * in_w + in_r * in_w + in_c;
                if (ds) {
                  weight_idx = out_ch * kernel_h * kernel_w +
                                in_ch * kernel_h * kernel_w +
                                ker_r * kernel_w + ker_c;
                } else  {
                  weight_idx = out_ch * in_channels * kernel_h * kernel_w +
                                in_ch * kernel_h * kernel_w +
                                ker_r * kernel_w + ker_c;
                }
                acc += x[input_idx] * w[weight_idx];
              }
            }
          }
          if (ds) break;
        }
        
        // if (output_idx == 9349) printf("acc: %d\n", acc);
        
        acc *= acc_scale;
        if (acc > INT32_MAX|| acc < INT32_MIN) {
          // printf("AccOver");
        }
        acc += (1 << (shift-1));
        acc >>= shift;
        // quantize to uint8
        // if (ds == 0)
        //   acc = acc < 0 ? 0 : acc;  // ReLu
        // else
        acc = acc < -output_zero_point ? -output_zero_point : acc;
        
        acc = acc > 255-output_zero_point ? 255-output_zero_point : acc;
        out[output_idx++] = acc;
      }
    }
  }
}

void maxpool2d(int32_t *x, int32_t *out,
               int channels, int in_h, int in_w,
               int kernel_h, int kernel_w) {
  // stride
  int stride_h = kernel_h;
  int stride_w = kernel_w;
  int output_h = 1 + (in_h - kernel_h) / stride_h;
  int output_w = 1 + (in_w - kernel_w) / stride_w;

  for (int ch = 0; ch < channels; ++ch) {
    for (int out_r = 0; out_r < output_h; ++out_r) {
      for (int out_c = 0; out_c < output_w; ++out_c) {
        int32_t max_pick = INT32_MIN;
        
        for (int ker_r = 0; ker_r < kernel_h; ++ker_r) {
          for (int ker_c = 0; ker_c < kernel_w; ++ker_c) {
            int in_r = out_r * stride_h + ker_r;
            int in_c = out_c * stride_w + ker_c;
            if (in_r >= 0 && in_r < in_h && in_c >= 0 && in_c < in_w) {
              int input_idx = ch * in_h * in_w + in_r * in_w + in_c;
              if (x[input_idx] > max_pick) max_pick = x[input_idx];
            }
          }
        }

        int output_idx = ch * output_h * output_w + out_r * output_w + out_c;
        out[output_idx] = max_pick;
      }
    }
  }
}

void avgpool2d(int32_t *x, int32_t *out,
               int channels, int in_h, int in_w,
               int kernel_h, int kernel_w) {
  // stride
  int stride_h = kernel_h;
  int stride_w = kernel_w;
  int output_h = 1 + (in_h - kernel_h) / stride_h;
  int output_w = 1 + (in_w - kernel_w) / stride_w;
  int rounding = kernel_h * kernel_w / 2;

  for (int ch = 0; ch < channels; ++ch) {
    for (int out_r = 0; out_r < output_h; ++out_r) {
      for (int out_c = 0; out_c < output_w; ++out_c) {
        int32_t total = 0;
        
        for (int ker_r = 0; ker_r < kernel_h; ++ker_r) {
          for (int ker_c = 0; ker_c < kernel_w; ++ker_c) {
            int in_r = out_r * stride_h + ker_r;
            int in_c = out_c * stride_w + ker_c;
            if (in_r >= 0 && in_r < in_h && in_c >= 0 && in_c < in_w) {
              int input_idx = ch * in_h * in_w + in_r * in_w + in_c;
              total += x[input_idx];
            }
          }
        }

        int output_idx = ch * output_h * output_w + out_r * output_w + out_c;
        out[output_idx] = (total + rounding) / (kernel_h * kernel_w);
      }
    }
  }
}

// void forward_pass(int16_t *x, int16_t *out) {
//   // alternate input and output buffers
//   int16_t buf1[512];
//   int16_t buf2[512];
//   linear(x, fc1_weights, fc1_bias, buf1, fc1_acc_scale, fc1_shift, fc1_output_zero_point, 624, 512);
//   // relu(buf1, 512);
//   linear(buf1, fc2_weights, fc2_bias, buf2, fc2_acc_scale, fc2_shift, fc2_output_zero_point, 512, 256);
//   // relu(buf2, 256);
//   linear(buf2, fc3_weights, fc3_bias, buf1, fc3_acc_scale, fc3_shift, fc3_output_zero_point, 256, 64);
//   // relu(buf1, 64);
//   linear(buf1, fc4_weights, fc4_bias, buf2, fc4_acc_scale, fc4_shift, fc4_output_zero_point, 64, 1);
//   *out = buf2[0];
// }

void forward_pass2(int32_t *x, int32_t *out) {
  // alternate input and output buffers
  int32_t buf1[20000]; // out of conv2b 32 * 26 * 12
  int32_t buf2[20000]; // out of conv1 16 * 52 * 12
  conv2d(x, conv1_weights, conv1_bias, buf1,
         conv1_acc_scale, conv1_shift, conv1_output_zero_point,
         52, 12, 1, 32, 3, 3, 1, 1, 0);
  relu(buf1, 52*12*32);
  maxpool2d(buf1, buf2, 32, 52, 12, 2, 1);
  conv2d(buf2, conv2_weights, conv2_bias, buf1,
         conv2_acc_scale, conv2_shift, conv2_output_zero_point,
         26, 12, 32, 64, 3, 3, 1, 1, 0);
  relu(buf1, 26*12*64);
  maxpool2d(buf1, buf2, 64, 26, 12, 2, 2);
  conv2d(buf2, conv3_weights, conv3_bias, buf1,
         conv3_acc_scale, conv3_shift, conv3_output_zero_point,
         13, 6, 64, 128, 3, 3, 1, 1, 0);
  relu(buf1, 13*6*128);
  avgpool2d(buf1, buf2, 128, 13, 6, 13, 6);
  linear(buf2, fc1_weights, fc1_bias, buf1, fc1_acc_scale, fc1_shift, fc1_output_zero_point, 128, 1);
  *out = buf1[0];
}


int main() {
  int32_t out = 0;
  // printf("Starting...\n");
  clock_t begin, end;
  double time_spent = 0.0;
  for (int i = 0; i < 100; i++) {
    begin = clock();
    forward_pass2(x[i], &out);
    end = clock();
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    // needs fc1 output scale
    double outf = (double)out * 0.03421637415885925;
    double threshold = 0.4;
    double prob = 1 / (1 + exp(-outf));
    printf("Out[%2d]: %4d Expected: %4d ", i, out, y[i]);
    printf("Prob: %5.2f%% Pred: %d\n", prob * 100, prob > threshold ? 1 : 0);
  }
  printf("Average Time Per Inference: %.3fms\n", time_spent / 100.0 * 1000.0);
  return 0;
}
