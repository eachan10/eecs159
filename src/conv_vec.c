#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "model_dump.h"
#include "test_vec.h"

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
          for (int ker_r = 0; ker_r < kernel_h; ++ker_r) {
            for (int ker_c = 0; ker_c < kernel_w; ++ker_c) {
              int in_r = out_r * stride + ker_r - padding;
              int in_c = out_c * stride + ker_c - padding;
              int weight_idx;
              
              // padding zeros as zero point
              if (in_r >= 0 && in_r < in_h && in_c >= 0 && in_c < in_w) {
                int input_idx = in_ch * in_h * in_w + in_r * in_w + in_c;
                weight_idx = out_ch * in_channels * kernel_h * kernel_w +
                              in_ch * kernel_h * kernel_w +
                              ker_r * kernel_w + ker_c;
                acc += x[input_idx] * w[weight_idx];
              }
            }
          }
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

// TO TEST REFORMATTED OUTPUT WITH REFORMATTED WEIGHTS
#define IDX_CONV(r, x, c, W, C) (((r)*(W) + (x))*(C) + (c))
#define IDX_WGHT(ic, kr, kc, oc, KH, KW, OC) (((((ic) * (KH)) + (kr)) * (KW) + (kc)) * (OC) + (oc))

void conv2d_vec(int32_t *x, int32_t *w, int32_t *b, int32_t *out,
                int32_t acc_scale, int32_t shift, int32_t output_zero_point,
                int in_h, int in_w,
                int in_channels, int out_channels,
                int kernel_h, int kernel_w,
                int padding) {

    int output_h = 1 + (in_h - kernel_h + 2 * padding);
    int output_w = 1 + (in_w - kernel_w + 2 * padding);

    // Initialize output buffer
    for (int i = 0; i < output_h * output_w; i++) {
        int base = i * out_channels;
    
        // Set all elements to corresponding bias
        for (int oc = 0; oc < out_channels; oc++) {
            out[base + oc] = b[oc];
        }
    }

    // Compute convolution
    for (int in_c = 0; in_c < in_w; in_c++) {
      
        for (int in_r = 0; in_r < in_h; in_r++) {

              for (int ic = 0; ic < in_channels; ic++) {

                int input_val = x[IDX_CONV(in_r, in_c, ic, in_w, in_channels)];
                if (IDX_CONV(in_r, in_c, ic, in_w, in_channels) == 0) {
                  printf("in0: %d\n", x[0]);
                }

                for (int kr = 0; kr < kernel_h; kr++) {

                    int out_r = in_r + padding - kr;
                    if (out_r < 0 || out_r >= output_h)
                        continue;

                    for (int kc = 0; kc < kernel_w; kc++) {

                        int out_c = in_c + padding - kc;
                        if (out_c < 0 || out_c >= output_w)
                            continue;

                        for (int oc = 0; oc < out_channels; oc++) {
                            if (IDX_CONV(out_r, out_c, oc, output_w, out_channels) == 0) {
                              printf("out0: %d\n", out[0]);
                              printf("weight: %d\n", w[IDX_WGHT(ic, kr, kc, 0, kernel_h, kernel_w, out_channels)]);
                              printf("input: %d\n", input_val);
                            }

                            out[IDX_CONV(out_r, out_c, oc, output_w, out_channels)] += 
                                input_val * w[IDX_WGHT(ic, kr, kc, oc, kernel_h, kernel_w, out_channels)];
                        }
                    }
                }
            }
        }
    }

    // Quantization
    for (int i = 0; i < output_h * output_w; i++) {

        int base = i * out_channels;
    
        for (int oc = 0; oc < out_channels; oc++) {
            // Scaling
            if(base+oc == 0) printf("acc0: %d\n", out[base+oc]);
            out[base + oc] *= acc_scale;
            if(base+oc == 0) printf("mul0: %d\n", out[base+oc]);
            out[base + oc] += (1 << (shift-1));
            if(base+oc == 0) printf("rnd0: %d\n", out[base+oc]);
            out[base + oc] >>= shift;
            if(base+oc == 0) printf("shft0: %d\n", out[base+oc]);
    
            // Saturation & Clamping
            out[base + oc] = out[base + oc] < -output_zero_point ? -output_zero_point : out[base + oc];
            out[base + oc] = out[base + oc] > 255-output_zero_point ? 255-output_zero_point : out[base + oc];
        }
    }
}

// TO TEST REFORMATTED OUTPUT WITH NORMAL WEIGHT FORMATTING
// #define IDX_IN(c, r, x, H, W) ((c)*(H)*(W) + (r)*(W) + (x))
// #define IDX_OUT(r, x, c, W, OC) (((r)*(W) + (x))*(OC) + (c))
// #define IDX_WGHT(oc, kr, kc, ic, KH, KW, IC) ((((oc) * IC + ic) * KH + kr) * KW + kc)

// void conv2d_vec(int32_t *x, int32_t *w, int32_t *b, int32_t *out,
//                 int32_t acc_scale, int32_t shift, int32_t output_zero_point,
//                 int in_h, int in_w,
//                 int in_channels, int out_channels,
//                 int kernel_h, int kernel_w,
//                 int padding) {

//     int output_h = 1 + (in_h - kernel_h + 2 * padding);
//     int output_w = 1 + (in_w - kernel_w + 2 * padding);

//     // Initialize output buffer
//     for (int i = 0; i < output_h * output_w; i++) {
//         int base = i * out_channels;
    
//         // Set all elements to corresponding bias
//         for (int oc = 0; oc < out_channels; oc++) {
//             out[base + oc] = b[oc];
//         }
//     }

//     // Compute convolution
//     for (int ic = 0; ic < in_channels; ic++) {
      
//         for (int in_r = 0; in_r < in_h; in_r++) {

//             for (int in_c = 0; in_c < in_w; in_c++) {

//                 int input_val = x[IDX_IN(ic, in_r, in_c, in_h, in_w)];

//                 for (int kr = 0; kr < kernel_h; kr++) {

//                     int out_r = in_r + padding - kr;
//                     if (out_r < 0 || out_r >= output_h)
//                         continue;

//                     for (int kc = 0; kc < kernel_w; kc++) {

//                         int out_c = in_c + padding - kc;
//                         if (out_c < 0 || out_c >= output_w)
//                             continue;

//                         for (int oc = 0; oc < out_channels; oc++) {
//                             int weight_idx = oc * in_channels * kernel_h * kernel_w +
//                               ic * kernel_h * kernel_w +
//                               kr * kernel_w + kc;
                              
//                             out[IDX_OUT(out_r, out_c, oc, output_w, out_channels)] += input_val * w[weight_idx];
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     // Quantization
//     for (int i = 0; i < output_h * output_w; i++) {

//         int base = i * out_channels;
    
//         for (int oc = 0; oc < out_channels; oc++) {
//             // Scaling
//             out[base + oc] *= acc_scale;
//             out[base + oc] += (1 << (shift-1));
//             out[base + oc] >>= shift;
    
//             // Saturation & Clamping
//             out[base + oc] = out[base + oc] < -output_zero_point ? -output_zero_point : out[base + oc];
//             out[base + oc] = out[base + oc] > 255-output_zero_point ? 255-output_zero_point : out[base + oc];
//         }
//     }
// }

int main() {
  int32_t buf1[20000];
  int32_t buf2[20000];

  conv2d(x[0], conv1_weights, conv1_bias, buf1,
         conv1_acc_scale, conv1_shift, conv1_output_zero_point,
         52, 12, 1, 32, 3, 3, 1, 1, 0);

  conv2d_vec(x[0], conv1_weights_vec, conv1_bias, buf2,
             conv1_acc_scale, conv1_shift, conv1_output_zero_point,
             52, 12, 1, 32, 3, 3, 1);

  int OH = 1 + (52 - 3 + 2 * 1);
  int OW = 1 + (12 - 3 + 2 * 1);
  int OC = 32;

  int errors = 0;

  for (int r = 0; r < OH; r++) {
      for (int c = 0; c < OW; c++) {
          for (int oc = 0; oc < OC; oc++) {

              int idx_new =
                  (r * OW + c) * OC + oc;

              int idx_old =
                  (oc * OH * OW) + (r * OW) + c;

              int32_t val_new = buf2[idx_new];
              int32_t val_old = buf1[idx_old];

              if (val_new != val_old) {
                  printf("Mismatch at (r=%d, c=%d, oc=%d): "
                        "new=%d old=%d\n",
                        r, c, oc, val_new, val_old);
                  errors++;
              }
          }
      }
  }

  if (errors == 0)
      printf("Outputs match!\n");
  else
      printf("Total mismatches: %d\n", errors);

  printf("\nint32_t buf1_array[%d] = {", OH * OW * OC);
  for (int i = 0; i < OH * OW * OC; i++) {
      if (i % 10 == 0) // 10 values per line
          printf("\n    ");
      printf("%d", buf2[i]);
      if (i < OH * OW * OC - 1)
          printf(", ");
  }
  printf("\n};\n");

  return 0;
}
