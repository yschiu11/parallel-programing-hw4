#ifndef __SHA256_HEADER__
#define __SHA256_HEADER__

#include <stddef.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"{
#endif  //__cplusplus

//--------------- DATA TYPES --------------
typedef unsigned int WORD;
typedef unsigned char BYTE;

typedef union _sha256_ctx{
	WORD h[8];
	BYTE b[32];
}SHA256;

//----------- UTILITY MACRO & FUNCTION --------
// (這個 _swap 宏是從 sha256.cu 複製過來的，以便 inline 函數使用)
#ifndef _swap
#define _swap(x, y) (((x)^=(y)), ((y)^=(x)), ((x)^=(y)))
#endif

// 新增：SHA256 最終結果的大小端轉換
static __host__ __device__ inline
void sha256_swap_endian(SHA256 *ctx)
{
	for(int i=0;i<32;i+=4)
	{
        _swap(ctx->b[i], ctx->b[i+3]);
        _swap(ctx->b[i+1], ctx->b[i+2]);
	}
}

//----------- FUNCTION DECLARATION --------
__host__ __device__
void sha256_transform(SHA256 *ctx, const BYTE *msg);

__host__ __device__
void sha256(SHA256 *ctx, const BYTE *msg, size_t len);


#ifdef __cplusplus
}
#endif  //__cplusplus

#endif  //__SHA256_HEADER__