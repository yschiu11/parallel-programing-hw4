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

// d_block:         指向裝置 (GPU) 記憶體的區塊頭
// d_target_hex:    指向裝置記憶體的目標困難度
// d_found_nonce:   指向裝置記憶體的旗標/結果變數。
//                  初始值為 0xFFFFFFFF，找到解的 thread 會用 atomicExch 寫入 nonce
// nonce_offset:    這批次 kernel 啟動的 nonce 起始偏移量
__global__
void solve_kernel(const HashBlock* d_block, const unsigned char* d_target_hex, unsigned int* d_found_nonce, unsigned long long nonce_offset)
{
    __shared__ HashBlock s_block;
    __shared__ unsigned int s_found_nonce;

    if (threadIdx.x == 0)
    {
        s_block = *d_block;
        s_found_nonce = *d_found_nonce;
    }

    __syncthreads();

    // 計算這個 thread 負責的 nonce
    unsigned long long nonce_ll = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x + nonce_offset;
    
    // 檢查是否超出 32-bit (0xFFFFFFFF) 的搜尋範圍
    if (nonce_ll > 0xFFFFFFFF)
    {
        return;
    }
    
    // 檢查是否已經有其他 thread 找到答案
    if (s_found_nonce != 0xFFFFFFFF)
    {
        return;
    }

    unsigned int nonce = (unsigned int)nonce_ll;

    // 將區塊頭複製到 thread 自己的 local memory
    HashBlock local_block = s_block;
    local_block.nonce = nonce;
    
    SHA256 sha256_ctx;
    
    // 執行雙重 SHA256 雜湊
    double_sha256(&sha256_ctx, (unsigned char*)&local_block, sizeof(local_block));
    
    // 比較結果
    if(little_endian_bit_comparison(sha256_ctx.b, d_target_hex, 32) < 0)  // sha256_ctx < target_hex
    {
        // 找到了！ 使用 atomicExch 確保只有一個 thread 能寫入結果
        // atomicExch 會回傳 d_found_nonce 的 "舊值"
        // 我們不在乎舊值，我們只想把我們的 nonce 寫上去
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
    // calculate target value from encoded difficulty which is encoded on "nbits"
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


    // ********** find nonce **************

    HashBlock *d_block;
    unsigned char *d_target_hex;
    unsigned int *d_found_nonce;
    unsigned int h_found_nonce = 0xFFFFFFFF;  // initial value for not found

    cudaMalloc((void**)&d_block, sizeof(HashBlock));
    cudaMalloc((void**)&d_target_hex, 32 * sizeof(unsigned char));
    cudaMalloc((void**)&d_found_nonce, sizeof(unsigned int));

    cudaMemcpy(d_block, &block, sizeof(HashBlock), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_hex, target_hex, 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_nonce, &h_found_nonce, sizeof(unsigned int), cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 128; // 每個 block 128 個 threads
    const int GRID_SIZE = 2048; // 每個 grid 2048 個 blocks
    unsigned long long batch_size = (unsigned long long)BLOCK_SIZE * GRID_SIZE;
    unsigned long long nonce_offset = 0;

    printf("Starting GPU search...\n");

    while (h_found_nonce == 0xFFFFFFFF) {
        solve_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_block, d_target_hex, d_found_nonce, nonce_offset);
        
        // 檢查是否有錯誤 (可選，但建議)
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
            break;
        }

        // 等待 kernel 完成 (可選，但檢查結果前需要)
        // cudaDeviceSynchronize(); 
        // 由於我們馬上要 DtoH 複製，這會隱含同步，所以上面那行可省
        
        // 將結果 (是否找到) 從 Device 複製回 Host
        cudaMemcpy(&h_found_nonce, d_found_nonce, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        // (可選) 印出進度
        // if (h_found_nonce == 0xFFFFFFFF) {
        //     printf("Checked nonces up to %llu, no solution found yet.\n", nonce_offset + batch_size - 1);
        // }

        // 檢查是否已搜尋完所有 2^32 個 nonce
        if (nonce_offset > 0xFFFFFFFF)
        {
            printf("Full nonce space searched, no solution found.\n");
            break;
        }
        
        // 準備下一批次的 nonce 偏移量
        nonce_offset += batch_size;
    }

    cudaFree(d_block);
    cudaFree(d_target_hex);
    cudaFree(d_found_nonce);
    
    SHA256 sha256_ctx;
    
    // for(block.nonce=0x00000000; block.nonce<=0xffffffff;++block.nonce)
    // {   
    //     //sha256d
    //     double_sha256(&sha256_ctx, (unsigned char*)&block, sizeof(block));
    //     if(block.nonce % 1000000 == 0)
    //     {
    //         printf("hash #%10u (big): ", block.nonce);
    //         print_hex_inverse(sha256_ctx.b, 32);
    //         printf("\n");
    //     }
        
    //     if(little_endian_bit_comparison(sha256_ctx.b, target_hex, 32) < 0)  // sha256_ctx < target_hex
    //     {
    //         printf("Found Solution!!\n");
    //         printf("hash #%10u (big): ", block.nonce);
    //         print_hex_inverse(sha256_ctx.b, 32);
    //         printf("\n\n");

    //         break;
    //     }
    // }

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
        // 迴圈結束了但還是沒找到 (理論上測試案例都會找到)
        printf("No solution found after searching.\n");
        // 為了讓程式能繼續，我們隨便設一個值
        memset(&sha256_ctx, 0, sizeof(sha256_ctx));
    }
    

    // print result

    //little-endian
    printf("hash(little): ");
    print_hex(sha256_ctx.b, 32);
    printf("\n");

    //big-endian
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
    std::chrono::duration<double> elapsed = end - start;
    printf("Elapsed time: %.6f seconds\n", elapsed.count());

    return 0;
}


// 11.665703 s