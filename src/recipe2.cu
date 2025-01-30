#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "vinegar.h"
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

int main(int argc, char **argv) {

    if(argc != 3) {
        ERROR_LOG("Invalid number of arguments \nUsage : %s <input image file> <output file name>\n", argv[0]);
        exit(255);
    }

    int width, height, channels;
    unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 0);
    size_t size  = width*height*channels;
    unsigned char *outputImage = (unsigned char*)malloc(size);

    unsigned char * image_d, * outputImage_d;

    dim3 threadsInBlock(16, 16, 1);
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

    cudaFree(outputImage_d);
    cudaFree(image_d);

    stbi_write_png(argv[2],width, height, channels, outputImage, width * channels);

    return(0);
}