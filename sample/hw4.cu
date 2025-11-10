//***********************************************************************************
// 2018.04.01 created by Zexlus1126
//
//    Example 002
// This is a simple demonstration on calculating merkle root from merkle branch 
// and solving a block (#286819) which the information is downloaded from Block Explorer 
//***********************************************************************************

#include <iostream>
#include <fstream>
#include <string>

#include <cstdio>
#include <cstring>

#include <cassert>
#include <cuda_runtime.h>

// 新增：為了 Host 端的 sleep (polling)
#include <thread>
#include <chrono>

#include "sha256.h"

////////////////////////   Block   /////////////////////

typedef struct _block
{
    unsigned int version;
    unsigned char prevhash[32];
    unsigned char merkle_root[32];
    unsigned int ntime;
    unsigned int nbits;
    unsigned int nonce;
}HashBlock;

typedef struct _BlockChunk2Data
{
    BYTE merkle_p4[4]; // merkle_root 的 [28..31]
    unsigned int ntime;
    unsigned int nbits;
} BlockChunk2Data;


////////////////////////   Utils   ///////////////////////

// (Utils 函數... decode, convert_string_to_little_endian_bytes, 
//  print_hex, print_hex_inverse, little_endian_bit_comparison, getline 
//  ... 均保持不變)
// ... (為節省篇幅，省略相同的 Utils 函數) ...
unsigned char decode(unsigned char c){
    switch(c){
        case 'a': return 0x0a;
        case 'b': return 0x0b;
        case 'c': return 0x0c;
        case 'd': return 0x0d;
        case 'e': return 0x0e;
        case 'f': return 0x0f;
        case '0' ... '9': return c-'0';
    }
    return 0;
}
void convert_string_to_little_endian_bytes(unsigned char* out, char *in, size_t string_len){
    assert(string_len % 2 == 0);
    size_t s = 0;
    size_t b = string_len/2-1;
    for(; s < string_len; s+=2, --b){
        out[b] = (unsigned char)(decode(in[s])<<4) + decode(in[s+1]);
    }
}
void print_hex(unsigned char* hex, size_t len){
    for(int i=0;i<len;++i) printf("%02x", hex[i]);
}
void print_hex_inverse(unsigned char* hex, size_t len){
    for(int i=len-1;i>=0;--i) printf("%02x", hex[i]);
}
__host__ __device__
int little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len){
    for(int i=byte_len-1;i>=0;--i){
        if(a[i] < b[i]) return -1;
        else if(a[i] > b[i]) return 1;
    }
    return 0;
}
void getline(char *str, size_t len, FILE *fp){
    int i=0;
    while( i<len && (str[i] = fgetc(fp)) != EOF && str[i++] != '\n');
    str[len-1] = '\0';
}


////////////////////////   Hash   ///////////////////////
static __host__ __device__
void double_sha256(SHA256 *sha256_ctx, unsigned char *bytes, size_t len)
{
    SHA256 tmp;
    sha256(&tmp, (BYTE*)bytes, len);
    sha256(sha256_ctx, (BYTE*)&tmp, sizeof(tmp));
}


////////////////////   Merkle Root   /////////////////////
// (calc_merkle_root 函數保持不變)
// ... (為節省篇幅，省略相同的 calc_merkle_root 函數) ...
void calc_merkle_root(unsigned char *root, int count, char **branch){
    size_t total_count = count;
    unsigned char *raw_list = new unsigned char[(total_count+1)*32];
    unsigned char **list = new unsigned char*[total_count+1];
    for(int i=0;i<total_count; ++i){
        list[i] = raw_list + i * 32;
        convert_string_to_little_endian_bytes(list[i], branch[i], 64);
    }
    list[total_count] = raw_list + total_count*32;
    while(total_count > 1){
        int i, j;
        if(total_count % 2 == 1){
            memcpy(list[total_count], list[total_count-1], 32);
        }
        for(i=0, j=0;i<total_count;i+=2, ++j){
            double_sha256((SHA256*)list[j], list[i], 64);
        }
        total_count = j;
    }
    memcpy(root, list[0], 32);
    delete[] raw_list;
    delete[] list;
}


////////////////////   CUDA Kernel   /////////////////////

// =================================================================
//
//    新：使用 "Mid-state" + "Grid-Stride Loop" 優化的 Kernel
//
// =================================================================
__global__
void solve_kernel_midstate(const SHA256* d_midstate,             // pre-calculated Mid-state
                           const BlockChunk2Data* d_chunk2_data, // 第 2 個 chunk 的固定資料
                           const unsigned char* d_target_hex,    // 目標值
                           unsigned int* d_found_nonce)          // 找到的 nonce
                           // (移除 nonce_offset)
{
    // --- 1. 使用共享記憶體 (優化 #1) ---
    __shared__ SHA256 s_midstate;
    __shared__ BlockChunk2Data s_chunk2_data;
    __shared__ unsigned int s_found_nonce;

    if (threadIdx.x == 0)
    {
        s_midstate = *d_midstate;
        s_chunk2_data = *d_chunk2_data;
        s_found_nonce = *d_found_nonce;
    }
    __syncthreads();

    // --- 2. 計算 Grid-Stride Loop 參數 ---
    unsigned long long start_nonce = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long grid_stride = (unsigned long long)gridDim.x * blockDim.x;

    BYTE m[64];
    BYTE m2[64];

    // prefill m with fixed data (without nonce)
    memset(m, 0, 64);
    memcpy(m, s_chunk2_data.merkle_p4, 4);   // merkle root last 4 bytes
    memcpy(m + 4, &s_chunk2_data.ntime, 4);  // ntime
    memcpy(m + 8, &s_chunk2_data.nbits, 4);  // nbits
    m[16] = 0x80;  // padding
    unsigned long long L_640 = 640; // length
    m[63] = (BYTE)L_640; 
    m[62] = (BYTE)(L_640 >> 8); 
    m[56] = (BYTE)(L_640 >> 56);

    // prefill m2 with fixed data (without hash from round 1)
    memset(m2, 0, 64);
    m2[32] = 0x80;  // padding
    unsigned long long L_256 = 256; // length
    m2[63] = (BYTE)L_256;
    m2[62] = (BYTE)(L_256 >> 8);

    // Grid-Stride Loop
    unsigned int i = 0;
    for (unsigned long long nonce_ll = start_nonce; 
         nonce_ll <= 0xFFFFFFFF; // 搜尋整個 32-bit 空間
         nonce_ll += grid_stride, ++i) {
        // check if found before expensive hash
        if ((i & 16384) == 0) {
            if (threadIdx.x == 0)
                s_found_nonce = *d_found_nonce;

            __syncthreads();  // make sure each thread reads updated s_found_nonce
            
            if (s_found_nonce != 0xFFFFFFFF)
                return;  // exit from all threads in the block
        }

        unsigned int nonce = (unsigned int)nonce_ll;

        // --- 4. 執行 SHA-256 (Round 1, 續) ---
        SHA256 ctx_round1 = s_midstate; 
         
        memcpy(m + 12, &nonce, 4);  // only write 4 bytes of changeable nonce        
        
        sha256_transform(&ctx_round1, m);
        sha256_swap_endian(&ctx_round1); 

        // --- 5. 執行 SHA-256 (Round 2, 全) ---
        SHA256 sha256_ctx;
        sha256_ctx.h[0] = 0x6a09e667;
        sha256_ctx.h[1] = 0xbb67ae85;
        sha256_ctx.h[2] = 0x3c6ef372;
        sha256_ctx.h[3] = 0xa54ff53a;
        sha256_ctx.h[4] = 0x510e527f;
        sha256_ctx.h[5] = 0x9b05688c;
        sha256_ctx.h[6] = 0x1f83d9ab;
        sha256_ctx.h[7] = 0x5be0cd19;
        
        memcpy(m2, ctx_round1.b, 32);  // only write 32 bytes of changeable hash from round 1
        
        sha256_transform(&sha256_ctx, m2);
        sha256_swap_endian(&sha256_ctx); 

        if (little_endian_bit_comparison(sha256_ctx.b, d_target_hex, 32) < 0) {
            atomicExch(d_found_nonce, nonce);
            // 找到後，這個 thread 可以離開，
            // 其他 thread 會在下一次迴圈的開頭檢查到 d_found_nonce 並離開
            // if (threadIdx.x == 0) 
            //     s_found_nonce = nonce;
        
            return;
        }
    }
}


void solve(FILE *fin, FILE *fout)
{
    // **** read data *****
    char version[9];
    char prevhash[65];
    char ntime[9];
    char nbits[9];
    int tx;
    char *raw_merkle_branch;
    char **merkle_branch;

    getline(version, 9, fin);
    getline(prevhash, 65, fin);
    getline(ntime, 9, fin);
    getline(nbits, 9, fin);
    fscanf(fin, "%d\n", &tx);
    printf("start hashing");

    raw_merkle_branch = new char [tx * 65];
    merkle_branch = new char *[tx];
    for(int i=0;i<tx;++i){
        merkle_branch[i] = raw_merkle_branch + i * 65;
        getline(merkle_branch[i], 65, fin);
        merkle_branch[i][64] = '\0';
    }
    unsigned char merkle_root[32];
    calc_merkle_root(merkle_root, tx, merkle_branch);
    printf("merkle root(little): "); print_hex(merkle_root, 32); printf("\n");
    printf("merkle root(big):    "); print_hex_inverse(merkle_root, 32); printf("\n");


    // (**** solve block **** 和 (Host) 準備 Mid-state/target... 部分不變)
    // ... (為節省篇幅，省略相同的 CPU 端準備程式碼) ...
    printf("Block info (big): \n");
    printf("  version:  %s\n", version);
    printf("  pervhash: %s\n", prevhash);
    printf("  merkleroot: "); print_hex_inverse(merkle_root, 32); printf("\n");
    printf("  nbits:    %s\n", nbits);
    printf("  ntime:    %s\n", ntime);
    printf("  nonce:    ???\n\n");

    HashBlock block;
    convert_string_to_little_endian_bytes((unsigned char *)&block.version, version, 8);
    convert_string_to_little_endian_bytes(block.prevhash,                  prevhash,    64);
    memcpy(block.merkle_root, merkle_root, 32);
    convert_string_to_little_endian_bytes((unsigned char *)&block.nbits,   nbits,     8);
    convert_string_to_little_endian_bytes((unsigned char *)&block.ntime,   ntime,     8);
    block.nonce = 0;
    
    unsigned int exp = block.nbits >> 24;
    unsigned int mant = block.nbits & 0xffffff;
    unsigned char target_hex[32] = {};
    unsigned int shift = 8 * (exp - 3);
    unsigned int sb = shift / 8;
    unsigned int rb = shift % 8;
    target_hex[sb    ] = (mant << rb);
    target_hex[sb + 1] = (mant >> (8-rb));
    target_hex[sb + 2] = (mant >> (16-rb));
    target_hex[sb + 3] = (mant >> (24-rb));
    printf("Target value (big): "); print_hex_inverse(target_hex, 32); printf("\n");

    SHA256 h_midstate;
    h_midstate.h[0] = 0x6a09e667; h_midstate.h[1] = 0xbb67ae85;
	h_midstate.h[2] = 0x3c6ef372; h_midstate.h[3] = 0xa54ff53a;
	h_midstate.h[4] = 0x510e527f; h_midstate.h[5] = 0x9b05688c;
	h_midstate.h[6] = 0x1f83d9ab; h_midstate.h[7] = 0x5be0cd19;
    BYTE chunk1[64];
    memcpy(chunk1, &block.version, 4);
    memcpy(chunk1 + 4, block.prevhash, 32);
    memcpy(chunk1 + 36, block.merkle_root, 28);
    sha256_transform(&h_midstate, chunk1);
    
    BlockChunk2Data h_chunk2_data;
    memcpy(h_chunk2_data.merkle_p4, block.merkle_root + 28, 4);
    h_chunk2_data.ntime = block.ntime;
    h_chunk2_data.nbits = block.nbits;
    
    SHA256 *d_midstate;
    BlockChunk2Data *d_chunk2_data;
    unsigned char *d_target_hex;
    unsigned int *d_found_nonce;
    unsigned int h_found_nonce = 0xFFFFFFFF;

    cudaMalloc((void**)&d_midstate, sizeof(SHA256));
    cudaMalloc((void**)&d_chunk2_data, sizeof(BlockChunk2Data));
    cudaMalloc((void**)&d_target_hex, 32 * sizeof(unsigned char));
    cudaMalloc((void**)&d_found_nonce, sizeof(unsigned int));

    cudaMemcpy(d_midstate, &h_midstate, sizeof(SHA256), cudaMemcpyHostToDevice);
    cudaMemcpy(d_chunk2_data, &h_chunk2_data, sizeof(BlockChunk2Data), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_hex, target_hex, 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_nonce, &h_found_nonce, sizeof(unsigned int), cudaMemcpyHostToDevice);

    // ********** find nonce (Grid-Stride Loop 優化版) **************

    // 啟動參數：
    // BLOCK_SIZE 仍然重要 (256 或 512 是好選擇)
    // GRID_SIZE 現在代表我們要啟動多少 *Block*。
    // 讓 (GRID_SIZE * BLOCK_SIZE) 遠大於 GPU 核心數，
    // 以確保 GPU 完全飽和。
    const int BLOCK_SIZE = 256; 
    const int GRID_SIZE = 1024 * 32; // 啟動 32k 個 blocks

    // (移除 batch_size 和 nonce_offset)

    printf("Starting GPU search (Mid-state + Grid-Stride Loop optimized)...\n");

    // --- 1. (Host) 啟動一次 Kernel ---
    // (注意：參數移除了 nonce_offset)
    solve_kernel_midstate<<<GRID_SIZE, BLOCK_SIZE>>>(
        d_midstate, d_chunk2_data, d_target_hex, d_found_nonce);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    // --- 2. (Host) 進入 Polling 迴圈 ---
    // (這取代了原來的 kernel 啟動迴圈)
    while (h_found_nonce == 0xFFFFFFFF) 
    {
        // 讓 CPU 睡一下，不要 100% 佔用
        // GPU 會在背景持續工作
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // 偶爾檢查一次 GPU 是否找到了答案
        cudaMemcpy(&h_found_nonce, d_found_nonce, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    // (迴圈結束代表 h_found_nonce != 0xFFFFFFFF, 也就是找到了)


    cudaFree(d_midstate);
    cudaFree(d_chunk2_data);
    cudaFree(d_target_hex);
    cudaFree(d_found_nonce);
    
    // (後續的 "Found Solution!!" 印出和驗證部分... 保持不變)
    // ... (為節省篇幅，省略相同的結果印出程式碼) ...
    SHA256 sha256_ctx;
    if (h_found_nonce != 0xFFFFFFFF){
        printf("Found Solution!!\n");
        block.nonce = h_found_nonce;
        double_sha256(&sha256_ctx, (unsigned char*)&block, sizeof(block));
        printf("hash #%10u (big): ", block.nonce);
        print_hex_inverse(sha256_ctx.b, 32);
        printf("\n\n");
    } else {
        printf("No solution found after searching.\n");
        memset(&sha256_ctx, 0, sizeof(sha256_ctx));
    }
    
    printf("hash(little): "); print_hex(sha256_ctx.b, 32); printf("\n");
    printf("hash(big):    "); print_hex_inverse(sha256_ctx.b, 32); printf("\n\n");
    for(int i=0;i<4;++i) fprintf(fout, "%02x", ((unsigned char*)&block.nonce)[i]);
    fprintf(fout, "\n");

    delete[] merkle_branch;
    delete[] raw_merkle_branch;
}

int main(int argc, char **argv)
{
    // (main 函數保持不變)
    if (argc != 3) {
        fprintf(stderr, "usage: cuda_miner <in> <out>\n");
    }
    FILE *fin = fopen(argv[1], "r");
    FILE *fout = fopen(argv[2], "w");

    int totalblock;
    fscanf(fin, "%d\n", &totalblock);
    fprintf(fout, "%d\n", totalblock);

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<totalblock;++i)
    {
        solve(fin, fout);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("Total time: %.6f seconds\n", duration.count());

    return 0;
}

// 12.82, 7.048524 6.936092 6.986965