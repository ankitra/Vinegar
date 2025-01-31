#include "vinegar.h"

void * recipie2_malloc(size_t);
void * recipie2_realloc(void *, size_t);
void   recipie2_free(void *);

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MALLOC(sz)           recipie2_malloc(sz)
#define STBI_REALLOC(p,newsz)     
#define STBI_REALLOC_SIZED(p,oldsz,newsz) recipie2_realloc((p),(oldsz),(newsz))
#define STBI_FREE(p)              recipie2_free(p)


#include "stb_image.h"
#include "stb_image_write.h"

#define INTERLEAVED(row, col, base, width, channels, offset) ROW_MAJOR(row, (channels)*(col) + offset, base, (width)*(channels), unsigned char)

__constant__ 
const float softFilterKernel[3][3] = 
{  {1.0/9.0,   1.0/9.0,   1.0/9.0},
   {1.0/9.0,   1.0/9.0,   1.0/9.0},
   {1.0/9.0,   1.0/9.0,   1.0/9.0}
}; 
__constant__
const int filterSize = 3;


__global__ 
void softenImageRGBKernel(unsigned char * image, unsigned char * imageOutput, \
                            int width, int height, int channels) {

    int row = THREAD_OFFSET(x);
    int col = THREAD_OFFSET(y);
    int offset = THREAD_OFFSET(z); // which channel are we acting upon? R/G/B?
    int i = 0,j=0;
    float pixel = 0.0; 
    float totalFilter = 0;
    const int kernel_lower =  -(filterSize/2), kernel_upper =  (filterSize/2), kernel_half = -kernel_lower;

    if(row >= height || col >= width)
        return;

    for(i = kernel_lower; i <= kernel_upper; i++) {
        for(j = kernel_lower; j <= kernel_upper; j++) {
            if(row+i >= 0 && row+i < height && col+j>=0 && col+j < width) { 
                pixel += (*(INTERLEAVED(row+i, col+j, image, width,channels, offset ))) * softFilterKernel[i+kernel_half][j+kernel_half];
                totalFilter += softFilterKernel[i+kernel_half][j+kernel_half];
            }
        }
    }

    *(INTERLEAVED(row, col, imageOutput, width,channels, offset )) = (unsigned char)(pixel / totalFilter);

}

void * recipie2_malloc(size_t size) {
    void * ptr;
    CHECKED_CUDA_API(cudaMallocHost(&ptr, size));
    return ptr;
}

void recipie2_free(void * ptr) {
    CHECKED_CUDA_API(cudaFreeHost(ptr));
}

void * recipie2_realloc(void * ptr, size_t oldsize, size_t size){
    void * newptr;
    CHECKED_CUDA_API(cudaMallocHost(&newptr, size));
    memcpy(newptr,ptr,oldsize);
    CHECKED_CUDA_API(cudaFreeHost(ptr));
    return newptr;
}

int main(int argc, char **argv) {

    if(argc != 3) {
        ERROR_LOG("Invalid number of arguments \nUsage : %s <input image file> <output file name>\n", argv[0]);
        exit(255);
    }

    int width, height, channels;
    unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 0);
    size_t size  = width*height*channels;
    unsigned char *outputImage;

    CHECKED_CUDA_API(cudaMallocHost(&outputImage, size));

    unsigned char * image_d, * outputImage_d;

    dim3 threadsInBlock(8, 8, 1);
    dim3 blocksInGrid((height + threadsInBlock.x -1)/threadsInBlock.x,
                      (width + threadsInBlock.y -1)/threadsInBlock.y, 
                      channels);

    if(image == NULL || outputImage == NULL) {
        ERROR_LOG("Can not read inputs. %s does not exists or too big for main memory\n", argv[1]);
        exit(255);
    }

    CHECKED_CUDA_API(cudaMalloc((void **) &image_d,size));
    CHECKED_CUDA_API(cudaMalloc((void **) &outputImage_d,size));

    CHECKED_CUDA_API(cudaMemcpy(image_d, image, size, cudaMemcpyHostToDevice));

    #define invocation softenImageRGBKernel<<<blocksInGrid, threadsInBlock>>>(image_d, outputImage_d, width, height, channels)
    DEBUG_LOG("Invoking Kernel");
    TIME_CUDA("Convolution Kernel", invocation);
    
    CHECKED_CUDA_API(cudaGetLastError());                                                                    
    
    CHECKED_CUDA_API(cudaDeviceSynchronize());

    CHECKED_CUDA_API(cudaMemcpy(outputImage, outputImage_d, size, cudaMemcpyDeviceToHost));

    CHECKED_CUDA_API(cudaFree(outputImage_d));
    CHECKED_CUDA_API(cudaFree(image_d));

    stbi_write_png(argv[2],width, height, channels, outputImage, width * channels);

    CHECKED_CUDA_API(cudaFreeHost(outputImage));
    CHECKED_CUDA_API(cudaFreeHost(image));

    return(0);
}