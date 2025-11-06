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

// 新增：這個結構體只儲存區塊頭第二個 chunk (64-byte) 中的
// 固定部分 (merkle root 最後 4 bytes, ntime, nbits)。
// 總共 4 + 4 + 4 = 12 bytes。
typedef struct _BlockChunk2Data
{
    BYTE merkle_p4[4]; // merkle_root 的 [28..31]
    unsigned int ntime;
    unsigned int nbits;
} BlockChunk2Data;


////////////////////////   Utils   ///////////////////////

//convert one hex-codec char to binary
unsigned char decode(unsigned char c)
{
    switch(c)
    {
        case 'a':
            return 0x0a;
        case 'b':
            return 0x0b;
        case 'c':
            return 0x0c;
        case 'd':
            return 0x0d;
        case 'e':
            return 0x0e;
        case 'f':
            return 0x0f;
        case '0' ... '9':
            return c-'0';
    }
    return 0;
}


// convert hex string to binary
//
// in: input string
// string_len: the length of the input string
//      '\0' is not included in string_len!!!
// out: output bytes array
void convert_string_to_little_endian_bytes(unsigned char* out, char *in, size_t string_len)
{
    assert(string_len % 2 == 0);

    size_t s = 0;
    size_t b = string_len/2-1;

    for(; s < string_len; s+=2, --b)
    {
        out[b] = (unsigned char)(decode(in[s])<<4) + decode(in[s+1]);
    }
}

// print out binary array (from highest value) in the hex format
void print_hex(unsigned char* hex, size_t len)
{
    for(int i=0;i<len;++i)
    {
        printf("%02x", hex[i]);
    }
}


// print out binar array (from lowest value) in the hex format
void print_hex_inverse(unsigned char* hex, size_t len)
{
    for(int i=len-1;i>=0;--i)
    {
        printf("%02x", hex[i]);
    }
}

__host__ __device__
int little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len)
{
    // compared from lowest bit
    for(int i=byte_len-1;i>=0;--i)
    {
        if(a[i] < b[i])
            return -1;
        else if(a[i] > b[i])
            return 1;
    }
    return 0;
}

void getline(char *str, size_t len, FILE *fp)
{

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


// calculate merkle root from several merkle branches
// root: output hash will store here (little-endian)
// branch: merkle branch  (big-endian)
// count: total number of merkle branch
void calc_merkle_root(unsigned char *root, int count, char **branch)
{
    size_t total_count = count; // merkle branch
    unsigned char *raw_list = new unsigned char[(total_count+1)*32];
    unsigned char **list = new unsigned char*[total_count+1];

    // copy each branch to the list
    for(int i=0;i<total_count; ++i)
    {
        list[i] = raw_list + i * 32;
        //convert hex string to bytes array and store them into the list
        convert_string_to_little_endian_bytes(list[i], branch[i], 64);
    }

    list[total_count] = raw_list + total_count*32;


    // calculate merkle root
    while(total_count > 1)
    {
        
        // hash each pair
        int i, j;

        if(total_count % 2 == 1)  //odd, 
        {
            memcpy(list[total_count], list[total_count-1], 32);
        }

        for(i=0, j=0;i<total_count;i+=2, ++j)
        {
            // this part is slightly tricky,
            //   because of the implementation of the double_sha256,
            //   we can avoid the memory begin overwritten during our sha256d calculation
            // double_sha:
            //     tmp = hash(list[0]+list[1])
            //     list[0] = hash(tmp)
            double_sha256((SHA256*)list[j], list[i], 64);
        }

        total_count = j;
    }

    memcpy(root, list[0], 32);

    delete[] raw_list;
    delete[] list;
}

////////////////////   CUDA Kernel   /////////////////////

// 舊的 Kernel (保留做對比)
__global__
void solve_kernel_original(const HashBlock* d_block, const unsigned char* d_target_hex, unsigned int* d_found_nonce, unsigned long long nonce_offset)
{
    // ... (原始程式碼) ...
}


// =================================================================
//
//    新：使用 "Mid-state" 優化的 Kernel
//
// =================================================================
__global__
void solve_kernel_midstate(const SHA256* d_midstate,             // 預先算好的 Mid-state
                           const BlockChunk2Data* d_chunk2_data, // 第 2 個 chunk 的固定資料
                           const unsigned char* d_target_hex,    // 目標值
                           unsigned int* d_found_nonce,          // 找到的 nonce
                           unsigned long long nonce_offset)     // Nonce 偏移
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

    // 檢查是否已有其他 thread 找到答案
    if (s_found_nonce != 0xFFFFFFFF)
    {
        return;
    }

    // --- 2. 計算 Nonce ---
    unsigned long long nonce_ll = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x + nonce_offset;
    if (nonce_ll > 0xFFFFFFFF)
    {
        return;
    }
    unsigned int nonce = (unsigned int)nonce_ll;

    // --- 3. 執行 SHA-256 (Round 1, 續) ---
    // (我們跳過了 Transform 1, 直接從 Mid-state 開始)
    
    // 從共享記憶體載入 Mid-state
    SHA256 ctx_round1 = s_midstate; 
    
    // 準備第 2 個 chunk (data[64...79] + padding)
    BYTE m[64];
    memset(m, 0, 64);

    memcpy(m, s_chunk2_data.merkle_p4, 4);  // merkle_root[28..31]
    memcpy(m + 4, &s_chunk2_data.ntime, 4);   // ntime
    memcpy(m + 8, &s_chunk2_data.nbits, 4);    // nbits
    memcpy(m + 12, &nonce, 4);                // nonce (變數!)

    m[16] = 0x80; // 80-byte 訊息的 padding (第 16 byte)

    // 總長度 (80 bytes * 8 bits = 640 bits)
    // SHA-256 規格要求長度為 64-bit Big Endian
    unsigned long long L = 640;
    m[63] = (BYTE)L;
	m[62] = (BYTE)(L >> 8);
	m[61] = (BYTE)(L >> 16);
	m[60] = (BYTE)(L >> 24);
	m[59] = (BYTE)(L >> 32);
	m[58] = (BYTE)(L >> 40);
	m[57] = (BYTE)(L >> 48);
	m[56] = (BYTE)(L >> 56);
    
    // 執行 Round 1 的第 2 次 Transform (Kernel Transform #1)
    sha256_transform(&ctx_round1, m);
    sha256_swap_endian(&ctx_round1); // 轉換為 Big Endian (標準 Hash 輸出)


    // --- 4. 執行 SHA-256 (Round 2, 全) ---
    
    // 這是 Round 2 的最終結果
    SHA256 sha256_ctx;
    
    // 初始化 Round 2 的 Hash 狀態
    sha256_ctx.h[0] = 0x6a09e667;
	sha256_ctx.h[1] = 0xbb67ae85;
	sha256_ctx.h[2] = 0x3c6ef372;
	sha256_ctx.h[3] = 0xa54ff53a;
	sha256_ctx.h[4] = 0x510e527f;
	sha256_ctx.h[5] = 0x9b05688c;
	sha256_ctx.h[6] = 0x1f83d9ab;
	sha256_ctx.h[7] = 0x5be0cd19;

    // 準備第 3 個 chunk (Round 1 的 32-byte 結果 + padding)
    BYTE m2[64];
    memset(m2, 0, 64);

    memcpy(m2, ctx_round1.b, 32); // Round 1 的結果
    m2[32] = 0x80; // 32-byte 訊息的 padding

    // 總長度 (32 bytes * 8 bits = 256 bits)
    L = 256;
    m2[63] = (BYTE)L;
	m2[62] = (BYTE)(L >> 8);
    // (其他 m2[56..61] 保持為 0)

    // 執行 Round 2 的 Transform (Kernel Transform #2)
    sha256_transform(&sha256_ctx, m2);
    sha256_swap_endian(&sha256_ctx); // 轉換為 Big Endian

    
    // --- 5. 比較結果 ---
    if(little_endian_bit_comparison(sha256_ctx.b, d_target_hex, 32) < 0)  // sha256_ctx < target_hex
    {
        atomicExch(d_found_nonce, nonce);
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
    for(int i=0;i<tx;++i)
    {
        merkle_branch[i] = raw_merkle_branch + i * 65;
        getline(merkle_branch[i], 65, fin);
        merkle_branch[i][64] = '\0';
    }

    // **** calculate merkle root ****

    unsigned char merkle_root[32];
    calc_merkle_root(merkle_root, tx, merkle_branch);

    printf("merkle root(little): ");
    print_hex(merkle_root, 32);
    printf("\n");

    printf("merkle root(big):    ");
    print_hex_inverse(merkle_root, 32);
    printf("\n");


    // **** solve block ****
    printf("Block info (big): \n");
    printf("  version:  %s\n", version);
    printf("  pervhash: %s\n", prevhash);
    printf("  merkleroot: "); print_hex_inverse(merkle_root, 32); printf("\n");
    printf("  nbits:    %s\n", nbits);
    printf("  ntime:    %s\n", ntime);
    printf("  nonce:    ???\n\n");

    HashBlock block;

    // convert to byte array in little-endian
    convert_string_to_little_endian_bytes((unsigned char *)&block.version, version, 8);
    convert_string_to_little_endian_bytes(block.prevhash,                  prevhash,    64);
    memcpy(block.merkle_root, merkle_root, 32);
    convert_string_to_little_endian_bytes((unsigned char *)&block.nbits,   nbits,     8);
    convert_string_to_little_endian_bytes((unsigned char *)&block.ntime,   ntime,     8);
    block.nonce = 0;
    
    
    // ********** calculate target value *********
    // (這部分不變)
    unsigned int exp = block.nbits >> 24;
    unsigned int mant = block.nbits & 0xffffff;
    unsigned char target_hex[32] = {};
    
    unsigned int shift = 8 * (exp - 3);
    unsigned int sb = shift / 8;
    unsigned int rb = shift % 8;
    
    // little-endian
    target_hex[sb    ] = (mant << rb);
    target_hex[sb + 1] = (mant >> (8-rb));
    target_hex[sb + 2] = (mant >> (16-rb));
    target_hex[sb + 3] = (mant >> (24-rb));
    
    
    printf("Target value (big): ");
    print_hex_inverse(target_hex, 32);
    printf("\n");


    // ********** find nonce (Mid-state 優化版) **************

    // --- 1. (Host) 預先計算 Mid-state ---
    SHA256 h_midstate;
    // 初始化
    h_midstate.h[0] = 0x6a09e667;
	h_midstate.h[1] = 0xbb67ae85;
	h_midstate.h[2] = 0x3c6ef372;
	h_midstate.h[3] = 0xa54ff53a;
	h_midstate.h[4] = 0x510e527f;
	h_midstate.h[5] = 0x9b05688c;
	h_midstate.h[6] = 0x1f83d9ab;
	h_midstate.h[7] = 0x5be0cd19;
    
    // 準備第 1 個 chunk (data[0...63])
    BYTE chunk1[64];
    memcpy(chunk1, &block.version, 4);
    memcpy(chunk1 + 4, block.prevhash, 32);
    memcpy(chunk1 + 36, block.merkle_root, 28); // 只複製 merkle_root 的前 28 bytes

    // 執行 Transform 1
    sha256_transform(&h_midstate, chunk1);
    
    // --- 2. (Host) 準備第 2 個 chunk 的固定資料 ---
    BlockChunk2Data h_chunk2_data;
    memcpy(h_chunk2_data.merkle_p4, block.merkle_root + 28, 4); // 複製 merkle_root 的後 4 bytes
    h_chunk2_data.ntime = block.ntime;
    h_chunk2_data.nbits = block.nbits;
    
    // --- 3. (Host) 準備 GPU 記憶體 ---
    // HashBlock *d_block; // 不再需要
    SHA256 *d_midstate;
    BlockChunk2Data *d_chunk2_data;
    unsigned char *d_target_hex;
    unsigned int *d_found_nonce;
    unsigned int h_found_nonce = 0xFFFFFFFF;  // initial value for not found

    // cudaMalloc((void**)&d_block, sizeof(HashBlock)); // 不再需要
    cudaMalloc((void**)&d_midstate, sizeof(SHA256));
    cudaMalloc((void**)&d_chunk2_data, sizeof(BlockChunk2Data));
    cudaMalloc((void**)&d_target_hex, 32 * sizeof(unsigned char));
    cudaMalloc((void**)&d_found_nonce, sizeof(unsigned int));

    // cudaMemcpy(d_block, &block, sizeof(HashBlock), cudaMemcpyHostToDevice); // 不再需要
    cudaMemcpy(d_midstate, &h_midstate, sizeof(SHA256), cudaMemcpyHostToDevice);
    cudaMemcpy(d_chunk2_data, &h_chunk2_data, sizeof(BlockChunk2Data), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_hex, target_hex, 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_nonce, &h_found_nonce, sizeof(unsigned int), cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 256; 
    const int GRID_SIZE = 1024; 
    unsigned long long batch_size = (unsigned long long)BLOCK_SIZE * GRID_SIZE;
    unsigned long long nonce_offset = 0;

    printf("Starting GPU search (Mid-state optimized)...\n");

    while (h_found_nonce == 0xFFFFFFFF) {
        // --- 4. (Host) 啟動新 Kernel ---
        solve_kernel_midstate<<<GRID_SIZE, BLOCK_SIZE>>>(
            d_midstate, d_chunk2_data, d_target_hex, d_found_nonce, nonce_offset);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
            break;
        }

        cudaMemcpy(&h_found_nonce, d_found_nonce, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        if (nonce_offset > 0xFFFFFFFF)
        {
            printf("Full nonce space searched, no solution found.\n");
            break;
        }
        
        nonce_offset += batch_size;
    }

    // --- 5. (Host) 釋放記憶體 ---
    // cudaFree(d_block); // 不再需要
    cudaFree(d_midstate);
    cudaFree(d_chunk2_data);
    cudaFree(d_target_hex);
    cudaFree(d_found_nonce);
    
    SHA256 sha256_ctx;
    
    // (CPU 端的舊迴圈已被 GPU 取代)

    if (h_found_nonce != 0xFFFFFFFF)
    {
        printf("Found Solution!!\n");
        
        // 用找到的 nonce 填入 host 端的 block
        block.nonce = h_found_nonce;
        
        // 在 host 上"再次"計算一次雜湊值，以便印出
        double_sha256(&sha256_ctx, (unsigned char*)&block, sizeof(block));
        
        printf("hash #%10u (big): ", block.nonce);
        print_hex_inverse(sha256_ctx.b, 32);
        printf("\n\n");
    }
    else
    {
        printf("No solution found after searching.\n");
        memset(&sha256_ctx, 0, sizeof(sha256_ctx));
    }
    
    // (印出結果部分不變)
    printf("hash(little): ");
    print_hex(sha256_ctx.b, 32);
    printf("\n");

    printf("hash(big):    ");
    print_hex_inverse(sha256_ctx.b, 32);
    printf("\n\n");

    for(int i=0;i<4;++i)
    {
        fprintf(fout, "%02x", ((unsigned char*)&block.nonce)[i]);
    }
    fprintf(fout, "\n");

    delete[] merkle_branch;
    delete[] raw_merkle_branch;
}

int main(int argc, char **argv)
{
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