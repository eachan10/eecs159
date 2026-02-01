#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "model_dump.h"
#include "test_vec.h"

// inputs: x, x_scale, x_zero_point, weight, weight_scale, weight_zero_point
//         bias, output_scale, output_zero_point

void linear(int16_t *x, int16_t *w, int32_t *b, int16_t *out,
            int32_t acc_scale, int32_t shift, int32_t output_zero_point,
            size_t input_dim, size_t output_dim) {
  for (size_t i = 0; i < output_dim; i++) {
    int32_t acc = b[i];
    for (size_t j = 0; j < input_dim; j++) {
      acc += x[j] * w[i*input_dim + j];
    }
    acc *= acc_scale;
    acc += (1 << (shift - 1));  // rounding
    acc >>= shift;
    // quantize to uint8
    // acc = acc < -output_zero_point ? -output_zero_point : acc;
    acc = acc < 0 ? 0 : acc;  // ReLu
    acc = acc > 255-output_zero_point ? 255-output_zero_point : acc;
    out[i] = acc;
  }
}

void relu(int16_t *x, size_t dim) {
  for (size_t i = 0; i < dim; i++) {
    x[i] = x[i] < 0 ? 0 : x[i];
  }
}

void forward_pass(int16_t *x, int16_t *out) {
  // alternate input and output buffers
  int16_t buf1[512];
  int16_t buf2[512];
  linear(x, fc1_weights, fc1_bias, buf1, fc1_acc_scale, fc1_shift, fc1_output_zero_point, 624, 512);
  // relu(buf1, 512);
  linear(buf1, fc2_weights, fc2_bias, buf2, fc2_acc_scale, fc2_shift, fc2_output_zero_point, 512, 256);
  // relu(buf2, 256);
  linear(buf2, fc3_weights, fc3_bias, buf1, fc3_acc_scale, fc3_shift, fc3_output_zero_point, 256, 64);
  // relu(buf1, 64);
  linear(buf1, fc4_weights, fc4_bias, buf2, fc4_acc_scale, fc4_shift, fc4_output_zero_point, 64, 1);
  *out = buf2[0];
}

int main() {
  int16_t out;
  printf("Starting...\n");
  for (size_t i = 0; i < 10; i++) {
    forward_pass(x[i], &out);
    printf("Out[%d]: %d Expected: %d\n", i, out, y[i]);
  }
  return 0;
}