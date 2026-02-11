#include <stdint.h>
#include <stdio.h>

#include "lin_test.h"
#include "model_lin.h"

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

int main() {
  // test
  int32_t out;
  for (int i = 0; i < 10; i++) {
    linear(x[i], fc1_weights, fc1_bias, &out,
           fc1_acc_scale, fc1_shift, fc1_output_zero_point,
           128, 1);
    printf("Expected: %d Actual: %d\n", y[i], out);
  }
}