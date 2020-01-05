#include <stdint.h>
#include <time.h>
#define patchSize 8
#define sharedSize 256
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


__global__
void calculateHistogram(uint8_t *imageData, int *countArray, int totalSize){
	__shared__ int private_histo[256];
	private_histo[threadIdx.x] = 0;
        int imageIndex = blockDim.x * blockIdx.x + threadIdx.x;
        if(imageIndex < totalSize){
		atomicAdd(&private_histo[imageData[imageIndex]], 1);
		__syncthreads();
		atomicAdd(&countArray[threadIdx.x], private_histo[threadIdx.x]);
	}
	
}

__global__
void maskImage(uint8_t *imageData, int *scannedArray, int totalSize){
        int imageIndex = blockDim.x * blockIdx.x + threadIdx.x;
        if(imageIndex < totalSize){
		imageData[imageIndex] = scannedArray[imageData[imageIndex]];
	}
}


__global__ void kogge_stone_scan(int *countArray, int *resultArray) {
  __shared__ int XY[2*sharedSize];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 256){
      XY[threadIdx.x] = countArray[i];
  }

  for (unsigned int stride = 1; stride <= sharedSize; stride *= 2) {
     __syncthreads();
     int index = (threadIdx.x+1)*stride*2 - 1;
     if (index < 2 * sharedSize){
         XY[index] += XY[index-stride];
     }
     
  }

  for (int stride = 2 * blockDim.x / 4; stride > 0; stride /= 2) {
    __syncthreads();

    int index = (threadIdx.x + 1) * 2 * stride - 1;

    if (index + stride < 2 * blockDim.x) {
      XY[index + stride] += XY[index];
    }
  }
  __syncthreads();
  if (i < 256) resultArray[i] = XY[threadIdx.x];
}


__global__
void calculateEqualization(int *scannedArray, int cdfmin, int totalSize){
	scannedArray[threadIdx.x] = int(	255 * (scannedArray[threadIdx.x] - cdfmin) / (totalSize - cdfmin)	);
}

int main(int argc, char ** argv) {
    /********** I use CUDA Event to calculate time but also use Clock to calculate total time ***************/
    clock_t total_start, total_end;
    total_start = clock();
    

    /********* Device Information by CUDA ***********************/
    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, 0);   
    printf("Device name: %s\n", deviceProp.name);
    printf("The maximum number of thread blocks: %d\n", deviceProp.maxGridSize[0] * deviceProp.maxGridSize[1]);
    printf("The maximum number of threads per block: %d\n\n", deviceProp.maxThreadsPerBlock);

    

    /*************** Read of Image Data, Image Width, Image Height by a library I found online ****************/
    int width, height, bpp;
    uint8_t* imageData = stbi_load(argv[1], &width, &height, &bpp, 1);
    //float* data = stbi_loadf(argv[1], &width, &height, &bpp, 1);
    printf("Width: %d Height: %d BPP: %d\n\n", width, height, bpp);



    /*************** CUDA Memory Allocation and Memory Copy for Image Data and also array to calculate histogram named arrayCount *****************/
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






    /******************* CUDA Grid Creation *******************************/
    int block = (width * height)/256 + ((width * height)/256.0 != 0.0);

    /******************* Calculating Histogram and Measure the Execution Time *****************/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    calculateHistogram<<<block, sharedSize>>>(rim, cim, width * height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaDeviceSynchronize();


    
    printf("Execution Time for Calculating Histogram: %f milliseconds\n\n", milliseconds);


    /****************** countArray is a small array (256 sized) so CPU is enough for its operation ************************/
    /****************** Here I calculate CDF ****************************/
    int scannedArray[256];
    int *sim;
    cudaMalloc( (void**)&sim, graysize );
    cudaMemcpy( sim, scannedArray , graysize, cudaMemcpyHostToDevice );

    int blockSize = 256/sharedSize + (256.0/sharedSize != 0.0);
    kogge_stone_scan<<<blockSize ,sharedSize>>>(cim, sim);
    cudaDeviceSynchronize();
    cudaMemcpy(scannedArray, sim, graysize, cudaMemcpyDeviceToHost);


    

    /****************** I store for range 0 to 255 but only use the range from minimum in image to maximum in image ****************/
    /****************** So minimum may not be in first place in array. So I search for it. It is found in few steps most probably ****************/
    int cdfmin;
    for(int i=0;i<256;i++){
		if(scannedArray[i] != 0){
			cdfmin = scannedArray[i];
			break;
		}
    }

    /****************** countArray is a small array (256 sized) so CPU is enough for its operation ************************/
    /****************** Here I calculate Equalization ****************************/
    calculateEqualization<<<1,256>>>(sim, cdfmin, width * height);  

  
    /***************** Memory Allocation and Memory Copy for my new countArray which is used for CDF and Equalization then ******************/
    /***************** Mask Image means convert value of image to new value of after CDF and Equalization ********************/
    cudaEventRecord(start);
    maskImage<<<block, sharedSize>>>(rim, sim, width * height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaDeviceSynchronize();
    

    cudaMemcpy(imageData, rim, imsize, cudaMemcpyDeviceToHost);

    printf("Execution Time for Masking Image: %f milliseconds\n\n", milliseconds);

    /***************** Here create new image after CDF and Equalization **************/
    /***************** You can give PNG or PGM files as input. But target image file format is PNG ************************/
    /***************** So after compile run should be like ./main file.pgm file2.png ************************/
    stbi_write_png(argv[2], width, height, 1, imageData, width*1);
    
    total_end = clock();
    float total_time = ((double) (total_end - total_start)) / (CLOCKS_PER_SEC/1000);
    printf("Total Time: %f milliseconds\n", total_time);
    return 0;
}


 

