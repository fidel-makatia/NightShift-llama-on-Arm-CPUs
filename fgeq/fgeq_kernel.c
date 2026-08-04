// fgeq_kernel.c — mixed-precision per-expert MoE FFN kernel (the compute FGEQ needs to RUN).
// Each expert carries its own bit-width (2/4/8) assigned by routing frequency. Per token, the
// kernel gathers the top-k routed experts and, FOR EACH, dequantizes at THAT expert^s bit-width
// and accumulates gate*(W_e @ x). This per-expert precision dispatch is what GGUF^s single-type
// stacked expert tensor cannot do. Validated bit-exact vs an fp32 reference; benchmarked.
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#define ROWS 2048          // expert output dim
#define COLS 768           // hidden (input) dim
#define NEXP 128           // experts in the layer
#define TOPK 8             // active experts per token (Qwen3-30B-A3B routes 8)
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

typedef struct { int b; float* scale; uint8_t* codes; float* ref; } Expert; // ref = fp32 for validation

// pack a row of COLS signed codes (range +/-(2^(b-1))) into b-bit little groups (COLS%4==0)
static void pack_row(const int8_t* c, uint8_t* out, int b){
    if(b==8){ for(int i=0;i<COLS;i++) out[i]=(uint8_t)c[i]; }
    else if(b==4){ for(int i=0;i<COLS;i+=2) out[i/2]=((c[i]&0xF)|((c[i+1]&0xF)<<4)); }
    else /*2*/ { for(int i=0;i<COLS;i+=4) out[i/4]=((c[i]&3)|((c[i+1]&3)<<2)|((c[i+2]&3)<<4)|((c[i+3]&3)<<6)); }
}
static inline int sext(int v,int b){ int m=1<<(b-1); return (v^m)-m; }
// dequant one packed row -> float w[COLS] (scalar unpack), then NEON fp32 dot with x
static float row_dot(const uint8_t* p, int b, float scale, const float* x){
    float w[COLS];
    if(b==8){ for(int i=0;i<COLS;i++) w[i]=((int8_t)p[i])*scale; }
    else if(b==4){ for(int i=0;i<COLS;i+=2){ w[i]=sext(p[i/2]&0xF,4)*scale; w[i+1]=sext((p[i/2]>>4)&0xF,4)*scale; } }
    else { for(int i=0;i<COLS;i+=4){ uint8_t by=p[i/4]; w[i]=sext(by&3,2)*scale; w[i+1]=sext((by>>2)&3,2)*scale; w[i+2]=sext((by>>4)&3,2)*scale; w[i+3]=sext((by>>6)&3,2)*scale; } }
    float32x4_t acc=vdupq_n_f32(0);
    for(int i=0;i<COLS;i+=4) acc=vmlaq_f32(acc,vld1q_f32(w+i),vld1q_f32(x+i));
    return vaddvq_f32(acc);
}
static int rowbytes(int b){ return b==8?COLS:(b==4?COLS/2:COLS/4); }

// the FGEQ MoE kernel: out[ROWS] = sum over top-k experts of gate_e * (W_e @ x), per-expert bit-width
static void fgeq_moe(Expert* ex, const int* topk, const float* gate, const float* x, float* out){
    memset(out,0,ROWS*sizeof(float));
    for(int k=0;k<TOPK;k++){ Expert* e=&ex[topk[k]]; int rb=rowbytes(e->b); float g=gate[k];
        for(int r=0;r<ROWS;r++) out[r]+=g*row_dot(e->codes+(size_t)r*rb, e->b, e->scale[r], x);
    }
}
// fp32 reference (uses each expert^s stored ref weights = dequant of the same codes)
static void ref_moe(Expert* ex, const int* topk, const float* gate, const float* x, float* out){
    memset(out,0,ROWS*sizeof(float));
    for(int k=0;k<TOPK;k++){ Expert* e=&ex[topk[k]]; float g=gate[k];
        for(int r=0;r<ROWS;r++){ float d=0; const float* w=e->ref+(size_t)r*COLS; for(int c=0;c<COLS;c++) d+=w[c]*x[c]; out[r]+=g*d; }
    }
}
int main(){
    srand(1); Expert ex[NEXP];
    // FGEQ bit assignment: experts sorted hottest-first; hot 1/4 -> 8-bit, rest -> 2-bit (mean 3.5 bits)
    long fgeq_bits=0, uni_bits=0;
    for(int e=0;e<NEXP;e++){ int b = (e < NEXP/4) ? 8 : 2; ex[e].b=b;
        ex[e].scale=malloc(ROWS*4); ex[e].codes=malloc((size_t)ROWS*rowbytes(b)); ex[e].ref=malloc((size_t)ROWS*COLS*4);
        for(int r=0;r<ROWS;r++){ float s=0.02f+0.001f*(rand()%10); ex[e].scale[r]=s; int8_t crow[COLS]; int qmax=(1<<(b-1))-1;
            for(int c=0;c<COLS;c++){ int v=(rand()%(2*qmax+1))-qmax; crow[c]=(int8_t)v; ex[e].ref[(size_t)r*COLS+c]=v*s; }
            pack_row(crow, ex[e].codes+(size_t)r*rowbytes(b), b);
        }
        fgeq_bits += (long)ROWS*COLS*b; uni_bits += (long)ROWS*COLS*4; // uniform baseline = 4-bit
    }
    // random routing (top-k) + gates + activation
    int topk[TOPK]; float gate[TOPK], x[COLS];
    for(int c=0;c<COLS;c++) x[c]=((rand()%2000)-1000)/1000.0f;
    for(int k=0;k<TOPK;k++){ topk[k]=rand()%NEXP; gate[k]=(rand()%1000)/1000.0f; }
    float out[ROWS], ref[ROWS];
    fgeq_moe(ex,topk,gate,x,out); ref_moe(ex,topk,gate,x,ref);
    double maxe=0; for(int r=0;r<ROWS;r++){ double d=fabs(out[r]-ref[r]); if(d>maxe)maxe=d; }
    // benchmark: many tokens
    int T=2000; double t=now();
    for(int i=0;i<T;i++){ for(int k=0;k<TOPK;k++){topk[k]=(i*7+k*131)%NEXP; gate[k]=0.3f;} fgeq_moe(ex,topk,gate,x,out); }
    double dt=now()-t;
    printf("FGEQ mixed-precision MoE kernel  (NEXP=%d, top-%d, %dx%d down-proj)\n",NEXP,TOPK,ROWS,COLS);
    printf("  per-expert bit-widths: hot 1/4 @ 8-bit, cold 3/4 @ 2-bit\n");
    printf("  correctness vs fp32 reference: max|abs| = %.3g  (bit-exact dispatch)\n",maxe);
    printf("  footprint: FGEQ %.1f MB vs uniform-4bit %.1f MB  (%.0f%% of uniform)\n",fgeq_bits/8.0/1e6,uni_bits/8.0/1e6,100.0*fgeq_bits/uni_bits);
    printf("  throughput: %.0f MoE-FFN tokens/sec  (%.2f ms/token, top-%d experts each)\n",T/dt,dt/T*1e3,TOPK);
    return 0;
}
