// In-cache compute-ceiling microbench for the quant dot-products K3 (iq1_s) and
// Qwen (q4_K) actually use, vs q8_0. Data stays resident => measures the COMPUTE rate
// (GB/s of weight bytes a kernel can consume when memory is free). Compare to the
// ~48 GB/s effective rate K3 hits in real generation to see if compute is the bottleneck.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
typedef uint16_t ggml_half; typedef uint32_t ggml_half2;
#define QK_K 256
#define QK8_0 32
#define K_SCALE_SIZE 12
typedef struct { ggml_half d; uint8_t qs[QK_K/8]; uint16_t qh[QK_K/32]; } block_iq1_s;
typedef struct { ggml_half d; ggml_half dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; } block_q4_K;
typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;
extern void ggml_vec_dot_iq1_s_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q4_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q8_0_q8_0(int,float*,size_t,const void*,size_t,const void*,size_t,int);
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
#define N 4096
#define NBK (N/QK_K)
#define NB80 (N/QK8_0)
static void fill(void*p,size_t n){ unsigned char*b=p; for(size_t i=0;i<n;i++) b[i]=rand(); }
int main(){
  int reps=400000; srand(1);
  block_iq1_s *w1=aligned_alloc(64,sizeof(block_iq1_s)*NBK);
  block_q4_K  *w4=aligned_alloc(64,sizeof(block_q4_K)*NBK);
  block_q8_0  *w8=aligned_alloc(64,sizeof(block_q8_0)*NB80);
  block_q8_K  *y =aligned_alloc(64,sizeof(block_q8_K)*NBK);
  block_q8_0  *y8=aligned_alloc(64,sizeof(block_q8_0)*NB80);
  fill(w1,sizeof(block_iq1_s)*NBK); fill(w4,sizeof(block_q4_K)*NBK);
  fill(w8,sizeof(block_q8_0)*NB80); fill(y,sizeof(block_q8_K)*NBK); fill(y8,sizeof(block_q8_0)*NB80);
  // tame the scale fields so results are finite (timing unaffected either way)
  for(int i=0;i<NBK;i++){ y[i].d=0.01f; w1[i].d=0x2C00; w4[i].d=0x2C00; w4[i].dmin=0x2C00; }
  for(int i=0;i<NB80;i++){ w8[i].d=0x2C00; y8[i].d=0x2C00; }
  float s; volatile double sink=0; double t,dt,wb;
  t=now(); for(int r=0;r<reps;r++){ ggml_vec_dot_iq1_s_q8_K(N,&s,0,w1,0,y,0,1); sink+=s; } dt=now()-t;
  wb=(double)sizeof(block_iq1_s)*NBK;
  printf("iq1_s (K3)   : %6.1f ns/call  %6.2f GB/s-compute  %5.1f GFLOP/s  (%.1f B/row, %.3f bits/w)\n", dt/reps*1e9, wb*reps/dt/1e9, 2.0*N*reps/dt/1e9, wb, wb*8.0/N);
  t=now(); for(int r=0;r<reps;r++){ ggml_vec_dot_q4_K_q8_K(N,&s,0,w4,0,y,0,1); sink+=s; } dt=now()-t;
  wb=(double)sizeof(block_q4_K)*NBK;
  printf("q4_K (Qwen)  : %6.1f ns/call  %6.2f GB/s-compute  %5.1f GFLOP/s  (%.1f B/row, %.3f bits/w)\n", dt/reps*1e9, wb*reps/dt/1e9, 2.0*N*reps/dt/1e9, wb, wb*8.0/N);
  t=now(); for(int r=0;r<reps;r++){ ggml_vec_dot_q8_0_q8_0(N,&s,0,w8,0,y8,0,1); sink+=s; } dt=now()-t;
  wb=(double)sizeof(block_q8_0)*NB80;
  printf("q8_0 (ref)   : %6.1f ns/call  %6.2f GB/s-compute  %5.1f GFLOP/s  (%.1f B/row, %.3f bits/w)\n", dt/reps*1e9, wb*reps/dt/1e9, 2.0*N*reps/dt/1e9, wb, wb*8.0/N);
  printf("sink=%.3e\n",(double)sink);
  return 0;
}
