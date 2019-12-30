#include <stdint.h>
#define patchSize 8
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


__global__
void calculateHistogram(uint8_t *imageData, int *countArray, int width, int height){
	int imageCol  =   blockIdx.x * blockDim.x + threadIdx.x;
	int imageRow  = blockIdx.y * blockDim.y + threadIdx.y;
        int imageIndex = imageRow * width + imageCol;
	int value;
        if(imageCol < width && imageRow < height){
		value = imageData[imageIndex];
		atomicAdd(&countArray[value], 1);
	}
}

__global__
void maskImage(uint8_t *imageData, int *countArray, int width, int height){
	int imageCol  =   blockIdx.x * blockDim.x + threadIdx.x;
	int imageRow  = blockIdx.y * blockDim.y + threadIdx.y;
        int imageIndex = imageRow * width + imageCol;
	int value;
        if(imageCol < width && imageRow < height){
		imageData[imageIndex] = countArray[imageData[imageIndex]];
	}
}

void calculateCDF(int *countArray){
	for(int i=0;i<256;i++){
		if(i != 255){
			countArray[i+1] = countArray[i+1] + countArray[i];
		}		
        }
}

void calculateEqualization(int *countArray, int cdfmin,int width,int height){
	int mn = width * height; 
	for(int i=0;i<256;i++){
		countArray[i] = int(	255 * (countArray[i] - cdfmin) / (mn - cdfmin)	);
	}
}

int main(int argc, char ** argv) {
    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, 0);
    int mtpb = deviceProp.maxThreadsPerBlock;        

    printf("Device name: %s\n", deviceProp.name);
    printf("The maximum number of thread blocks: %d\n", deviceProp.maxGridSize[0] * deviceProp.maxGridSize[1]);
    printf("The maximum number of threads per block: %d\n", mtpb);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Read of image data
    int width, height, bpp;
    uint8_t* imageData = stbi_load(argv[1], &width, &height, &bpp, 1);
    //float* data = stbi_loadf(argv[1], &width, &height, &bpp, 1);
    printf("%d %d %d %d\n", imageData[0], imageData[10], imageData[100], imageData[1000]);
    printf("%d %d %d %d\n", imageData[0]+5, imageData[10]+5, imageData[100]+5, imageData[1000]+5);
    printf("Width: %d Height: %d BPP: %d\n", width, height, bpp);

    uint8_t *rim;
    const int imsize = width*height*sizeof(uint8_t);
    cudaMalloc( (void**)&rim, imsize );
    cudaMemcpy( rim, imageData , imsize, cudaMemcpyHostToDevice );
    
    int countArray[256];
    for(int i=0;i<256;i++){
        countArray[i] = 0;
    }

    int *cim;
    const int graysize = 256 * sizeof(int);
    cudaMalloc( (void**)&cim, graysize );
    cudaMemcpy( cim, countArray , graysize, cudaMemcpyHostToDevice );


    int grid_x = width;
    int grid_y = height;
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(patchSize, patchSize, 1);

    cudaEventRecord(start);
    calculateHistogram<<<grid, block>>>(rim, cim, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaDeviceSynchronize();
    cudaMemcpy(imageData, rim, imsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(countArray, cim, graysize, cudaMemcpyDeviceToHost);


    
    printf("Execution Time for Calculating Histogram: %f milliseconds\n", milliseconds);

    calculateCDF(countArray);

    int cdfmin;
    for(int i=0;i<256;i++){
		if(countArray[i] != 0){
			cdfmin = countArray[i];
			break;
		}
    }

    calculateEqualization(countArray, cdfmin, width, height);  

  
    
    cudaMalloc( (void**)&cim, graysize );
    cudaMemcpy( cim, countArray , graysize, cudaMemcpyHostToDevice );
 
    cudaEventRecord(start);
    maskImage<<<grid, block>>>(rim, cim, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaDeviceSynchronize();
    

    cudaMemcpy(imageData, rim, imsize, cudaMemcpyDeviceToHost);

    printf("Execution Time for Masking Image: %f milliseconds\n", milliseconds);

    stbi_write_png("image.png", width, height, 1, imageData, width*1);
    
    
    return 0;
}


 

