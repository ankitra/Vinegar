#include "vinegar.h"
#include "stb_image.h"
#include "stb_image_write.h"

#define INTERLEAVED_R(row, col, base, width) ROW_MAJOR(row, 3*(col),   base, width*3, 1, char)
#define INTERLEAVED_G(row, col, base, width) ROW_MAJOR(row, 3*(col)+1, base, width*3, 1, char)
#define INTERLEAVED_B(row, col, base, width) ROW_MAJOR(row, 3*(col)+2, base, width*3, 1, char)

__global__ 
void softenImageRGBKernel(char * image, char * imageOutput, int width, int height) {

    int row = THREAD_OFFSET(x);
    int col = THREAD_OFFSET(y);

}

int main(int argc, char **argv) {

    if(argc != 3) {
        fprintf(stderr, "Usage :\nrecipe2 <input image file> <output file name>");
        fflush(stderr);
        exit(255);
    }

    return(0);
}