#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
__global__
void image_blur(float *imageData, float *countArray, int width, int height){
	int imageCol  =   blockIdx.x * blockDim.x + threadIdx.x;
	int imageRow  = blockIdx.y * blockDim.y + threadIdx.y;
	if(imageCol <= width && imageRow <= height){
		printf("aaas");
	}
	

}

__global__
void print(){
        int imageCol  =   blockIdx.x * blockDim.x + threadIdx.x;
	int imageRow  = blockIdx.y * blockDim.y + threadIdx.y;
	printf("%d %d\n", imageCol, imageRow);
}

int main(int argc, char ** argv) {
    int width, height, bpp;

    uint8_t* rgb_image = stbi_load(argv[1], &width, &height, &bpp, 1);
    float* data = stbi_loadf(argv[1], &width, &height, &bpp, 1);
    printf("Width: %d Height: %d BPP: %d\n", width, height, bpp);
    printf("%d", rgb_image[1]);
    for(int i=0; i<30000; i++){
        rgb_image[i] = 255;
    }
    stbi_write_png("image.png", width, height, 1, rgb_image, width*1);
    
    cudaDeviceSynchronize();
    return 0;
}


 

