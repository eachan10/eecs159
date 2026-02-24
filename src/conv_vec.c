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
                  // printf("\nin0: %d\n", x[0]);
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
                            // if (IDX_CONV(out_r, out_c, oc, output_w, out_channels) == 0) {
                            //   printf("out0: %d\n", out[0]);
                            //   printf("weight: %d\n", w[IDX_WGHT(ic, kr, kc, 0, kernel_h, kernel_w, out_channels)]);
                            //   printf("input: %d\n", input_val);
                            // }

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
            // if(base+oc == 0) printf("acc0: %d\n", out[base+oc]);
            out[base + oc] *= acc_scale;
            // if(base+oc == 0) printf("mul0: %d\n", out[base+oc]);
            out[base + oc] += (1 << (shift-1));
            // if(base+oc == 0) printf("rnd0: %d\n", out[base+oc]);
            out[base + oc] >>= shift;
            // if(base+oc == 0) printf("shft0: %d\n", out[base+oc]);
    
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

// TO TEST CONVOLUTION ONLY
// int main() {
//   int32_t buf1[20000];
//   int32_t buf2[20000];

//   conv2d(x[0], conv1_weights, conv1_bias, buf1,
//          conv1_acc_scale, conv1_shift, conv1_output_zero_point,
//          52, 12, 1, 32, 3, 3, 1, 1, 0);

//   conv2d_vec(x[0], conv1_weights_vec, conv1_bias, buf2,
//              conv1_acc_scale, conv1_shift, conv1_output_zero_point,
//              52, 12, 1, 32, 3, 3, 1);

//   int OH = 1 + (52 - 3 + 2 * 1);
//   int OW = 1 + (12 - 3 + 2 * 1);
//   int OC = 32;

//   int errors = 0;

//   for (int r = 0; r < OH; r++) {
//       for (int c = 0; c < OW; c++) {
//           for (int oc = 0; oc < OC; oc++) {

//               int idx_new =
//                   (r * OW + c) * OC + oc;

//               int idx_old =
//                   (oc * OH * OW) + (r * OW) + c;

//               int32_t val_new = buf2[idx_new];
//               int32_t val_old = buf1[idx_old];

//               if (val_new != val_old) {
//                   printf("Mismatch at (r=%d, c=%d, oc=%d): "
//                         "new=%d old=%d\n",
//                         r, c, oc, val_new, val_old);
//                   errors++;
//               }
//           }
//       }
//   }

//   if (errors == 0)
//       printf("Outputs match!\n");
//   else
//       printf("Total mismatches: %d\n", errors);

//   printf("\nint32_t buf1_array[%d] = {", OH * OW * OC);
//   for (int i = 0; i < OH * OW * OC; i++) {
//       if (i % 10 == 0) // 10 values per line
//           printf("\n    ");
//       printf("%d", buf2[i]);
//       if (i < OH * OW * OC - 1)
//           printf(", ");
//   }
//   printf("\n};\n");

//   return 0;
// }

void relu(int32_t *x, int dim) {
  for (int i = 0; i < dim; i++) {
    x[i] = x[i] < 0 ? 0 : x[i];
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

void maxpool2d_vec(int32_t *x, int32_t *out,
                   int channels, int in_h, int in_w,
                   int kernel_h, int kernel_w) {
    // stride = kernel size
    int stride_h = kernel_h;
    int stride_w = kernel_w;
    int output_h = 1 + (in_h - kernel_h) / stride_h;
    int output_w = 1 + (in_w - kernel_w) / stride_w;

    // Initialize output buffer
    for (int i = 0; i < output_h * output_w; i++) {
        int base = i * channels;
    
        // Set all elements to corresponding bias
        for (int oc = 0; oc < channels; oc++) {
            out[base + oc] = 0;
        }
    }

    for (int out_r = 0; out_r < output_h; out_r++) {

        for (int out_c = 0; out_c < output_w; out_c++) {

            for (int kr = 0; kr < kernel_h; kr++) {

                int in_r = out_r * stride_h + kr;
                if (in_r >= in_h) continue;

                for (int kc = 0; kc < kernel_w; kc++) {

                    int in_c = out_c * stride_w + kc;
                    if (in_c >= in_w) continue;

                    // Vectorizable loop over channels
                    for (int oc = 0; oc < channels; oc++) {
                        int32_t val = x[IDX_CONV(in_r, in_c, oc, in_w, channels)];
                        int32_t *out_ptr = &out[IDX_CONV(out_r, out_c, oc, output_w, channels)];
                        
                        // if (IDX_CONV(out_r, out_c, oc, output_w, channels) == 0) {
                        //   printf("input: %d\n", x[0]);
                        //   printf("output: %d\n", out[0]);
                        // }

                        if (val > *out_ptr) *out_ptr = val;
                    }
                }
            }
        }
    }
}

// HELPER FUNCTION FOR WEIGHT REFORMATTING
void reformat_weights(
  const int32_t *src,  // [OC][IC][KH][KW]
  int32_t *dst,        // [IC][KH][KW][OC]
  int OC, int IC, int KH, int KW
) {
  for (int oc = 0; oc < OC; oc++)
      for (int ic = 0; ic < IC; ic++)
          for (int kh = 0; kh < KH; kh++)
              for (int kw = 0; kw < KW; kw++) {
                  int src_idx =
                      (((oc * IC + ic) * KH + kh) * KW + kw);
                  int dst_idx =
                      (((ic * KH + kh) * KW + kw) * OC + oc);
                  dst[dst_idx] = src[src_idx];
              }
}

// TO TEST STAGE 1 CONVOLUTION + POOLING ONLY
// int main() {
//   int32_t buf1[20000];
//   int32_t buf2[20000];
//   int32_t buf3[20000];

//   conv2d(x[0], conv1_weights, conv1_bias, buf1,
//          conv1_acc_scale, conv1_shift, conv1_output_zero_point,
//          52, 12, 1, 32, 3, 3, 1, 1, 0);
//   relu(buf1, 52*12*32);
//   maxpool2d(buf1, buf2, 32, 52, 12, 2, 1);

//   conv2d_vec(x[0], conv1_weights_vec, conv1_bias, buf1,
//              conv1_acc_scale, conv1_shift, conv1_output_zero_point,
//              52, 12, 1, 32, 3, 3, 1);
//   relu(buf1, 52*12*32);
//   maxpool2d_vec(buf1, buf3, 32, 52, 12, 2, 1);

//   int OH = 26;
//   int OW = 12;
//   int OC = 32;

//   int errors = 0;

//   for (int r = 0; r < OH; r++) {
//       for (int c = 0; c < OW; c++) {
//           for (int oc = 0; oc < OC; oc++) {

//               int idx_new =
//                   (r * OW + c) * OC + oc;

//               int idx_old =
//                   (oc * OH * OW) + (r * OW) + c;

//               int32_t val_new = buf3[idx_new];
//               int32_t val_old = buf2[idx_old];

//               if (val_new != val_old) {
//                   printf("Mismatch at (r=%d, c=%d, oc=%d): "
//                         "new=%d old=%d\n",
//                         r, c, oc, val_new, val_old);
//                   errors++;
//               }
//           }
//       }
//   }

//   if (errors == 0)
//       printf("Outputs match!\n");
//   else
//       printf("Total mismatches: %d\n", errors);

//   printf("\nint32_t y[%d] = {", OH * OW * OC);
//   for (int i = 0; i < OH * OW * OC; i++) {
//       if (i % 10 == 0) // 10 values per line
//           printf("\n    ");
//       printf("%d", buf3[i]);
//       if (i < OH * OW * OC - 1)
//           printf(", ");
//   }
//   printf("\n};\n");

//   // To generate transposed conv2 weights
//   int32_t new_weights[18432];
//   int IC = 32;
//   OC = 64;
//   int KH = 3;
//   int KW = 3;
//   reformat_weights(conv2_weights, new_weights, 64, 32, 3, 3);
//   printf("\nint32_t conv2_weights_vec[%d] = {", OC * IC * KH * KW);
//   for (int i = 0; i < OC * IC * KH * KW; i++) {
//       if (i % 16 == 0) // 16 values per line
//           printf("\n    ");
//       printf("%d", new_weights[i]);
//       if (i < OC * IC * KH * KW - 1)
//           printf(", ");
//   }
//   printf("\n};\n");

//   return 0;
// }

// TO TEST STAGE 1&2 CONVOLUTION + POOLING ONLY
// int main() {
//   int32_t buf1[20000];
//   int32_t buf2[20000];
//   int32_t buf3[20000];

//   conv2d(x[0], conv1_weights, conv1_bias, buf1,
//          conv1_acc_scale, conv1_shift, conv1_output_zero_point,
//          52, 12, 1, 32, 3, 3, 1, 1, 0);
//   relu(buf1, 52*12*32);
//   maxpool2d(buf1, buf2, 32, 52, 12, 2, 1);
//   conv2d(buf2, conv2_weights, conv2_bias, buf1,
//          conv2_acc_scale, conv2_shift, conv2_output_zero_point,
//          26, 12, 32, 64, 3, 3, 1, 1, 0);

//   conv2d_vec(x[0], conv1_weights_vec, conv1_bias, buf2,
//              conv1_acc_scale, conv1_shift, conv1_output_zero_point,
//              52, 12, 1, 32, 3, 3, 1);
//   relu(buf2, 52*12*32);
//   maxpool2d_vec(buf2, buf3, 32, 52, 12, 2, 1);
//   conv2d_vec(buf3, conv2_weights_vec, conv2_bias, buf2,
//              conv2_acc_scale, conv2_shift, conv2_output_zero_point,
//              26, 12, 32, 64, 3, 3, 1);

//   int OH = 26;
//   int OW = 12;
//   int OC = 64;

//   int errors = 0;

//   for (int r = 0; r < OH; r++) {
//       for (int c = 0; c < OW; c++) {
//           for (int oc = 0; oc < OC; oc++) {

//               int idx_new =
//                   (r * OW + c) * OC + oc;

//               int idx_old =
//                   (oc * OH * OW) + (r * OW) + c;

//               int32_t val_new = buf2[idx_new];
//               int32_t val_old = buf1[idx_old];

//               if (val_new != val_old) {
//                   printf("Mismatch at (r=%d, c=%d, oc=%d): "
//                         "new=%d old=%d\n",
//                         r, c, oc, val_new, val_old);
//                   errors++;
//               }
//           }
//       }
//   }

//   if (errors == 0)
//       printf("Outputs match!\n");
//   else
//       printf("Total mismatches: %d\n", errors);

//   printf("\nint32_t y[%d] = {", OH * OW * OC);
//   for (int i = 0; i < OH * OW * OC; i++) {
//       if (i % 16 == 0) // 16 values per line
//           printf("\n    ");
//       printf("%d", buf2[i]);
//       if (i < OH * OW * OC - 1)
//           printf(", ");
//   }
//   printf("\n};\n");

//   return 0;
// }

// TO TEST STAGE 1&2 CONVOLUTION + 1&2 POOLING ONLY
// int main() {
//   int32_t buf1[20000];
//   int32_t buf2[20000];
//   int32_t buf3[20000];

//   conv2d(x[0], conv1_weights, conv1_bias, buf1,
//          conv1_acc_scale, conv1_shift, conv1_output_zero_point,
//          52, 12, 1, 32, 3, 3, 1, 1, 0);
//   relu(buf1, 52*12*32);
//   maxpool2d(buf1, buf2, 32, 52, 12, 2, 1);
//   conv2d(buf2, conv2_weights, conv2_bias, buf1,
//          conv2_acc_scale, conv2_shift, conv2_output_zero_point,
//          26, 12, 32, 64, 3, 3, 1, 1, 0);
//   relu(buf1, 26*12*64);
//   maxpool2d(buf1, buf2, 64, 26, 12, 2, 2);

//   conv2d_vec(x[0], conv1_weights_vec, conv1_bias, buf1,
//              conv1_acc_scale, conv1_shift, conv1_output_zero_point,
//              52, 12, 1, 32, 3, 3, 1);
//   relu(buf1, 52*12*32);
//   maxpool2d_vec(buf1, buf3, 32, 52, 12, 2, 1);
//   conv2d_vec(buf3, conv2_weights_vec, conv2_bias, buf1,
//              conv2_acc_scale, conv2_shift, conv2_output_zero_point,
//              26, 12, 32, 64, 3, 3, 1);
//   relu(buf1, 26*12*64);
//   maxpool2d_vec(buf1, buf3, 64, 26, 12, 2, 2);

//   int OH = 13;
//   int OW = 6;
//   int OC = 64;

//   int errors = 0;

//   for (int r = 0; r < OH; r++) {
//       for (int c = 0; c < OW; c++) {
//           for (int oc = 0; oc < OC; oc++) {

//               int idx_new =
//                   (r * OW + c) * OC + oc;

//               int idx_old =
//                   (oc * OH * OW) + (r * OW) + c;

//               int32_t val_new = buf3[idx_new];
//               int32_t val_old = buf2[idx_old];

//               if (val_new != val_old) {
//                   printf("Mismatch at (r=%d, c=%d, oc=%d): "
//                         "new=%d old=%d\n",
//                         r, c, oc, val_new, val_old);
//                   errors++;
//               }
//           }
//       }
//   }

//   if (errors == 0)
//       printf("Outputs match!\n");
//   else
//       printf("Total mismatches: %d\n", errors);

//   printf("\nint32_t y[%d] = {", OH * OW * OC);
//   for (int i = 0; i < OH * OW * OC; i++) {
//       if (i % 16 == 0) // 16 values per line
//           printf("\n    ");
//       printf("%d", buf3[i]);
//       if (i < OH * OW * OC - 1)
//           printf(", ");
//   }
//   printf("\n};\n");

//   // To generate transposed conv2 weights
//   int32_t new_weights[73728];
//   int IC = 64;
//   OC = 128;
//   int KH = 3;
//   int KW = 3;
//   reformat_weights(conv3_weights, new_weights, 128, 64, 3, 3);
//   printf("\nint32_t conv3_weights_vec[%d] = {", OC * IC * KH * KW);
//   for (int i = 0; i < OC * IC * KH * KW; i++) {
//       if (i % 16 == 0) // 16 values per line
//           printf("\n    ");
//       printf("%d", new_weights[i]);
//       if (i < OC * IC * KH * KW - 1)
//           printf(", ");
//   }
//   printf("\n};\n");

//   return 0;
// }

// TO TEST STAGE 1,2,3 CONVOLUTION + 1&2 POOLING ONLY
// int main() {
//   int32_t buf1[20000];
//   int32_t buf2[20000];
//   int32_t buf3[20000];

//   conv2d(x[0], conv1_weights, conv1_bias, buf1,
//          conv1_acc_scale, conv1_shift, conv1_output_zero_point,
//          52, 12, 1, 32, 3, 3, 1, 1, 0);
//   relu(buf1, 52*12*32);
//   maxpool2d(buf1, buf2, 32, 52, 12, 2, 1);
//   conv2d(buf2, conv2_weights, conv2_bias, buf1,
//          conv2_acc_scale, conv2_shift, conv2_output_zero_point,
//          26, 12, 32, 64, 3, 3, 1, 1, 0);
//   relu(buf1, 26*12*64);
//   maxpool2d(buf1, buf2, 64, 26, 12, 2, 2);
//   conv2d(buf2, conv3_weights, conv3_bias, buf1,
//          conv3_acc_scale, conv3_shift, conv3_output_zero_point,
//          13, 6, 64, 128, 3, 3, 1, 1, 0);

//   conv2d_vec(x[0], conv1_weights_vec, conv1_bias, buf2,
//              conv1_acc_scale, conv1_shift, conv1_output_zero_point,
//              52, 12, 1, 32, 3, 3, 1);
//   relu(buf2, 52*12*32);
//   maxpool2d_vec(buf2, buf3, 32, 52, 12, 2, 1);
//   conv2d_vec(buf3, conv2_weights_vec, conv2_bias, buf2,
//              conv2_acc_scale, conv2_shift, conv2_output_zero_point,
//              26, 12, 32, 64, 3, 3, 1);
//   relu(buf2, 26*12*64);
//   maxpool2d_vec(buf2, buf3, 64, 26, 12, 2, 2);
//   conv2d_vec(buf3, conv3_weights_vec, conv3_bias, buf2,
//              conv3_acc_scale, conv3_shift, conv3_output_zero_point,
//              13, 6, 64, 128, 3, 3, 1);

//   int OH = 13;
//   int OW = 6;
//   int OC = 128;

//   int errors = 0;

//   for (int r = 0; r < OH; r++) {
//       for (int c = 0; c < OW; c++) {
//           for (int oc = 0; oc < OC; oc++) {

//               int idx_new =
//                   (r * OW + c) * OC + oc;

//               int idx_old =
//                   (oc * OH * OW) + (r * OW) + c;

//               int32_t val_new = buf2[idx_new];
//               int32_t val_old = buf1[idx_old];

//               if (val_new != val_old) {
//                   printf("Mismatch at (r=%d, c=%d, oc=%d): "
//                         "new=%d old=%d\n",
//                         r, c, oc, val_new, val_old);
//                   errors++;
//               }
//           }
//       }
//   }

//   if (errors == 0)
//       printf("Outputs match!\n");
//   else
//       printf("Total mismatches: %d\n", errors);

//   printf("\nint32_t y[%d] = {", OH * OW * OC);
//   for (int i = 0; i < OH * OW * OC; i++) {
//       if (i % 16 == 0) // 16 values per line
//           printf("\n    ");
//       printf("%d", buf2[i]);
//       if (i < OH * OW * OC - 1)
//           printf(", ");
//   }
//   printf("\n};\n");

//   return 0;
// }

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

void avgpool2d_vec(int32_t *x, int32_t *out,
                   int32_t acc_scale, int32_t shift,
                   int channels, int in_h, int in_w,
                   int kernel_h, int kernel_w) {
    // stride
    int stride_h = kernel_h;
    int stride_w = kernel_w;
    int output_h = 1 + (in_h - kernel_h) / stride_h;
    int output_w = 1 + (in_w - kernel_w) / stride_w;
    int kernel_area = kernel_h * kernel_w;
    int rounding = kernel_area / 2;

    for (int out_r = 0; out_r < output_h; out_r++) {

        for (int out_c = 0; out_c < output_w; out_c++) {

            int32_t sum[channels];
            for (int c = 0; c < channels; c++)
                sum[c] = 0;

            for (int kr = 0; kr < kernel_h; kr++) {
                int in_r = out_r * stride_h + kr;
                if (in_r >= in_h) continue;

                for (int kc = 0; kc < kernel_w; kc++) {
                    int in_c = out_c * stride_w + kc;
                    if (in_c >= in_w) continue;

                    for (int c = 0; c < channels; c++) {
                        sum[c] += x[IDX_CONV(in_r, in_c, c, in_w, channels)];
                    }
                }
            }

            for (int c = 0; c < channels; c++) {
                // out[IDX_CONV(out_r, out_c, c, output_w, channels)] =
                //   (sum[c] + rounding) / kernel_area;
                out[IDX_CONV(out_r, out_c, c, output_w, channels)] = sum[c];
                out[IDX_CONV(out_r, out_c, c, output_w, channels)] += rounding;
                out[IDX_CONV(out_r, out_c, c, output_w, channels)] *= acc_scale;
                out[IDX_CONV(out_r, out_c, c, output_w, channels)] >>= shift;
            }
        }
    }
}

// TO TEST STAGE 1,2,3 CONVOLUTION + 1,2,3 POOLING ONLY
// int main() {
//   int32_t buf1[20000];
//   int32_t buf2[20000];
//   int32_t buf3[20000];

//   conv2d(x[0], conv1_weights, conv1_bias, buf1,
//          conv1_acc_scale, conv1_shift, conv1_output_zero_point,
//          52, 12, 1, 32, 3, 3, 1, 1, 0);
//   relu(buf1, 52*12*32);
//   maxpool2d(buf1, buf2, 32, 52, 12, 2, 1);
//   conv2d(buf2, conv2_weights, conv2_bias, buf1,
//          conv2_acc_scale, conv2_shift, conv2_output_zero_point,
//          26, 12, 32, 64, 3, 3, 1, 1, 0);
//   relu(buf1, 26*12*64);
//   maxpool2d(buf1, buf2, 64, 26, 12, 2, 2);
//   conv2d(buf2, conv3_weights, conv3_bias, buf1,
//          conv3_acc_scale, conv3_shift, conv3_output_zero_point,
//          13, 6, 64, 128, 3, 3, 1, 1, 0);
//   relu(buf1, 13*6*128);
//   avgpool2d(buf1, buf2, 128, 13, 6, 13, 6);

//   conv2d_vec(x[0], conv1_weights_vec, conv1_bias, buf1,
//              conv1_acc_scale, conv1_shift, conv1_output_zero_point,
//              52, 12, 1, 32, 3, 3, 1);
//   relu(buf1, 52*12*32);
//   maxpool2d_vec(buf1, buf3, 32, 52, 12, 2, 1);
//   conv2d_vec(buf3, conv2_weights_vec, conv2_bias, buf1,
//              conv2_acc_scale, conv2_shift, conv2_output_zero_point,
//              26, 12, 32, 64, 3, 3, 1);
//   relu(buf1, 26*12*64);
//   maxpool2d_vec(buf1, buf3, 64, 26, 12, 2, 2);
//   conv2d_vec(buf3, conv3_weights_vec, conv3_bias, buf1,
//              conv3_acc_scale, conv3_shift, conv3_output_zero_point,
//              13, 6, 64, 128, 3, 3, 1);
//   relu(buf1, 13*6*128);
//   avgpool2d_vec(buf1, buf3, avgpool_acc_scale, avgpool_shift,
//                 128, 13, 6, 13, 6);

//   int OH = 1;
//   int OW = 1;
//   int OC = 128;

//   int errors = 0;

//   for (int r = 0; r < OH; r++) {
//       for (int c = 0; c < OW; c++) {
//           for (int oc = 0; oc < OC; oc++) {

//               int idx_new =
//                   (r * OW + c) * OC + oc;

//               int idx_old =
//                   (oc * OH * OW) + (r * OW) + c;

//               int32_t val_new = buf3[idx_new];
//               int32_t val_old = buf2[idx_old];

//               if (val_new != val_old) {
//                   printf("Mismatch at (r=%d, c=%d, oc=%d): "
//                         "new=%d old=%d\n",
//                         r, c, oc, val_new, val_old);
//                   errors++;
//               }
//           }
//       }
//   }

//   if (errors == 0)
//       printf("Outputs match!\n");
//   else
//       printf("Total mismatches: %d\n", errors);

//   printf("\nint32_t y[%d] = {", OH * OW * OC);
//   for (int i = 0; i < OH * OW * OC; i++) {
//       if (i % 16 == 0) // 16 values per line
//           printf("\n    ");
//       printf("%d", buf3[i]);
//       if (i < OH * OW * OC - 1)
//           printf(", ");
//   }
//   printf("\n};\n");

//   return 0;
// }

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

void linear_to_one(int32_t *x, int32_t *w, int32_t b, int32_t *out,
                   int32_t acc_scale, int32_t shift, int32_t output_zero_point,
                   int input_dim) {
  int32_t acc = b;
  for (int j = 0; j < input_dim; j++) {
    acc += x[j] * w[j];
  }

  acc *= acc_scale;
  acc += (1 << (shift - 1));  // rounding
  acc >>= shift;
  // quantize to uint8
  acc = acc < -output_zero_point ? -output_zero_point : acc;
  // acc = acc < 0 ? 0 : acc;  // ReLu
  acc = acc > 255-output_zero_point ? 255-output_zero_point : acc;
  *out = acc;
}

// TO TEST ENTIRE CNN ONLY
int main() {
  int32_t buf1[20000];
  int32_t buf2[20000];
  int32_t buf3[20000];

  // // Scalar CNN
  // conv2d(x[0], conv1_weights, conv1_bias, buf1,
  //        conv1_acc_scale, conv1_shift, conv1_output_zero_point,
  //        52, 12, 1, 32, 3, 3, 1, 1, 0);
  // relu(buf1, 52*12*32);
  // maxpool2d(buf1, buf2, 32, 52, 12, 2, 1);
  // conv2d(buf2, conv2_weights, conv2_bias, buf1,
  //        conv2_acc_scale, conv2_shift, conv2_output_zero_point,
  //        26, 12, 32, 64, 3, 3, 1, 1, 0);
  // relu(buf1, 26*12*64);
  // maxpool2d(buf1, buf2, 64, 26, 12, 2, 2);
  // conv2d(buf2, conv3_weights, conv3_bias, buf1,
  //        conv3_acc_scale, conv3_shift, conv3_output_zero_point,
  //        13, 6, 64, 128, 3, 3, 1, 1, 0);
  // relu(buf1, 13*6*128);
  // avgpool2d(buf1, buf2, 128, 13, 6, 13, 6);
  // linear(buf2, fc1_weights, fc1_bias, buf1, fc1_acc_scale, fc1_shift, fc1_output_zero_point, 128, 3);

  // Pseudo-Vectorized CNN
  for (int i = 0; i < 100; i++) {
    conv2d_vec(x[i], conv1_weights, conv1_bias, buf2,
              conv1_acc_scale, conv1_shift, conv1_output_zero_point,
              52, 12, 1, 32, 3, 3, 1);
    relu(buf2, 52*12*32);
    maxpool2d_vec(buf2, buf3, 32, 52, 12, 2, 1);
    conv2d_vec(buf3, conv2_weights, conv2_bias, buf2,
              conv2_acc_scale, conv2_shift, conv2_output_zero_point,
              26, 12, 32, 64, 3, 3, 1);
    relu(buf2, 26*12*64);
    maxpool2d_vec(buf2, buf3, 64, 26, 12, 2, 2);
    conv2d_vec(buf3, conv3_weights, conv3_bias, buf2,
              conv3_acc_scale, conv3_shift, conv3_output_zero_point,
              13, 6, 64, 128, 3, 3, 1);
    relu(buf2, 13*6*128);
    avgpool2d_vec(buf2, buf3, avgpool_acc_scale, avgpool_shift,
                  128, 13, 6, 13, 6);
    linear_to_one(buf3, fc1_weights, fc1_bias[0], buf2, fc1_acc_scale, fc1_shift, fc1_output_zero_point, 128);
    linear_to_one(buf3, &fc1_weights[128], fc1_bias[1], &buf2[1], fc1_acc_scale, fc1_shift, fc1_output_zero_point, 128);
    linear_to_one(buf3, &fc1_weights[128*2], fc1_bias[2], &buf2[2], fc1_acc_scale, fc1_shift, fc1_output_zero_point, 128);

    printf("Golden: %d %d %d ", y[i*3], y[i*3+1], y[i*3+2]);
    printf("Vec: %d %d %d\n", buf2[0], buf2[1], buf2[2]);

  }
  return 0;
}
