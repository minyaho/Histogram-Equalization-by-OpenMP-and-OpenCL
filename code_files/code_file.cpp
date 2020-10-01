#include <iostream>
#include <CL/cl.h>
#include <fstream>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#define  IMG_W 2048
#define  IMG_H 2048
#define  IMG_SIZE 4194304
#define maxThread 8
#define CHUNKSIZE 20
using namespace std;

//------------------------------------------------------------------------------
unsigned char header[54] = {  0x42, 0x4d, 0, 0, 0, 0, 0, 0, 0, 0,54, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

typedef struct IMG_obj {
      unsigned char           *InputSrc,  *InputData;       // Input Image: 2048x2048
	  unsigned char			  *sR, *sG, *sB, *sY;                // Input data
	  unsigned short		  IMGW, IMGH;	            // Input Size
} IMGObj, *pImgObj;
IMGObj 		  imgobj;

int RReadIMG(IMGObj *IObj, const char *fname);
int SSaveIMG(IMGObj *IObj, const char *fname);
int SSaveIMGX(unsigned char *ptr_r, unsigned char *ptr_g, unsigned char *ptr_b, const char *fname);
int create_buffer(IMGObj *IObj);
int delete_buffer(IMGObj *IObj);
int IMG_TransOMP(IMGObj *IObj);
int IMG_Trans(IMGObj *IObj);
int IMG_TransOpenCL(IMGObj *IObj);

int create_buffer(IMGObj *IObj) {
	IObj->IMGW		=  IMG_H;
	IObj->IMGH		=  IMG_W;
	IObj->InputSrc   = (unsigned char*)malloc(sizeof(unsigned char)*IMG_SIZE*3+54);
	IObj->InputData  = IObj->InputSrc+54;

	IObj->sR   	= (unsigned char*)malloc(sizeof(unsigned char)*IMG_SIZE);  // source image
	IObj->sG  	= (unsigned char*)malloc(sizeof(unsigned char)*IMG_SIZE);
	IObj->sB  	= (unsigned char*)malloc(sizeof(unsigned char)*IMG_SIZE);
	IObj->sY  	= (unsigned char*)malloc(sizeof(unsigned char)*IMG_SIZE);
	return 0;
}
int delete_buffer(IMGObj *IObj) {
    free(IObj->InputSrc);
    free(IObj->sR);     free(IObj->sG);     free(IObj->sB);     free(IObj->sY);
    return 0;
}
//==============================================================================
int convertToString(const char *filename, std::string& s)
{
    size_t size;
    char*  str;

    std::fstream f(filename, (std::fstream::in | std::fstream::binary));

    if(f.is_open())
    {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);

        str = new char[size+1];
        if(!str)
        {
            f.close();
            return NULL;
        }

        f.read(str, fileSize);
        f.close();
        str[size] = '\0';

        s = str;
        delete[] str;
        return 0;
    }
    printf("Error: Failed to open file %s\n", filename);
    return 1;
}

int main(void)
{
	create_buffer(&imgobj);
	printf("1 read image\n");
	RReadIMG(&imgobj, "5.bmp");

	printf("2 image_cpu_trans\n");
	IMG_Trans(&imgobj);
	printf("3 save cpu image\n");
	SSaveIMG(&imgobj, "Output_CPU.BMP");

	printf("4 image_omp_trans\n");
	IMG_TransOMP(&imgobj);
	printf("5 save omp image\n");
	SSaveIMG(&imgobj, "Output_OMP.BMP");

    printf("6 image_openCL_trans\n");
	IMG_TransOpenCL(&imgobj);
    printf("7 save openCL image\n");

    delete_buffer(&imgobj);

    return 0;
}

int IMG_TransOpenCL(IMGObj *IObj){
	int x,y,index=0, off=0;

    for(y = 0; y < IMG_H; y++)
    {
        for(x = 0; x < IMG_W; x++)
        {
            IObj->sR[off]	 = IObj->InputSrc[index+2];
            IObj->sG[off]  = IObj->InputSrc[index+1];
            IObj->sB[off]  = IObj->InputSrc[index+0];
            off++;
            index += 3;
        }
    }

    index = 0;
    int hist_R[256] = {0};  int hist_G[256] = {0};  int hist_B[256] = {0};
    for(y=1; y!=IMG_H-1; y++){
        index = y*IMG_W;
        for(x=1; x!=IMG_W; x++){
            hist_R[IObj->sR[index]]++;
            hist_G[IObj->sG[index]]++;
            hist_B[IObj->sB[index]]++;
            index++;
        }
    }

    int cdf_R[256] = {0}; int cdf_G[256] = {0}; int cdf_B[256] = {0};
    int v,k;
    for(v=0; v<256; v++){

        if(hist_R[v]>0){
            for(k=v; k<256; k++) cdf_R[k]+=hist_R[v];
        }
        if(hist_G[v]>0){
            for(k=v; k<256; k++) cdf_G[k]+=hist_G[v];
        }
        if(hist_B[v]>0){
            for(k=v; k<256; k++) cdf_B[k]+=hist_B[v];
        }
    }

    int minCdfValue_R = 2048*2048; int minCdfValue_G = 2048*2048; int minCdfValue_B = 2048*2048;
    for(int v=0; v<256; v++){
        if((cdf_R[v] != 0) && (cdf_R[v] < minCdfValue_R)) minCdfValue_R = cdf_R[v];
        if((cdf_G[v] != 0) && (cdf_G[v] < minCdfValue_G)) minCdfValue_G = cdf_G[v];
        if((cdf_B[v] != 0) && (cdf_B[v] < minCdfValue_B)) minCdfValue_B = cdf_B[v];
    }

	cl_uint status;
	cl_platform_id platform;
	status = clGetPlatformIDs(1, &platform, NULL);

	cl_device_id device;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,1 ,&device, NULL);
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
	cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

	cl_mem clbufR = clCreateBuffer(context, CL_MEM_READ_WRITE, IMG_SIZE*sizeof(cl_uchar), NULL, NULL);
	status = clEnqueueWriteBuffer(queue, clbufR, 1, 0, IMG_SIZE*sizeof(cl_uchar), imgobj.sR,0 ,0 ,0);

	cl_mem clbufG = clCreateBuffer(context, CL_MEM_READ_WRITE, IMG_SIZE*sizeof(cl_uchar), NULL, NULL);
	status = clEnqueueWriteBuffer(queue, clbufG, 1, 0, IMG_SIZE*sizeof(cl_uchar), imgobj.sG,0 ,0 ,0);

	cl_mem clbufB = clCreateBuffer(context, CL_MEM_READ_WRITE, IMG_SIZE*sizeof(cl_uchar), NULL, NULL);
	status = clEnqueueWriteBuffer(queue, clbufB, 1, 0, IMG_SIZE*sizeof(cl_uchar), imgobj.sB,0 ,0 ,0);

	cl_mem clMinCdfValR = clCreateBuffer(context, CL_MEM_READ_ONLY, 1*sizeof(int), NULL, NULL);
	status = clEnqueueWriteBuffer(queue, clMinCdfValR, 1, 0, 1*sizeof(int), &minCdfValue_R,0 ,0 ,0);

	cl_mem clMinCdfValB = clCreateBuffer(context, CL_MEM_READ_ONLY, 1*sizeof(int), NULL, NULL);
	status = clEnqueueWriteBuffer(queue, clMinCdfValB, 1, 0, 1*sizeof(int), &minCdfValue_G,0 ,0 ,0);

    cl_mem clMinCdfValG = clCreateBuffer(context, CL_MEM_READ_ONLY, 1*sizeof(int), NULL, NULL);
	status = clEnqueueWriteBuffer(queue, clMinCdfValG, 1, 0, 1*sizeof(int), &minCdfValue_B,0 ,0 ,0);

	cl_mem clCdfR = clCreateBuffer(context, CL_MEM_READ_ONLY, 256*sizeof(int), NULL, NULL);
	status = clEnqueueWriteBuffer(queue, clCdfR, 1, 0, 256*sizeof(int), cdf_R,0 ,0 ,0);

	cl_mem clCdfG = clCreateBuffer(context, CL_MEM_READ_ONLY, 256*sizeof(int), NULL, NULL);
	status = clEnqueueWriteBuffer(queue, clCdfG, 1, 0, 256*sizeof(int), cdf_G,0 ,0 ,0);

    cl_mem clCdfB = clCreateBuffer(context, CL_MEM_READ_ONLY, 256*sizeof(int), NULL, NULL);
	status = clEnqueueWriteBuffer(queue, clCdfB, 1, 0, 256*sizeof(int), cdf_B,0 ,0 ,0);

	const char * filename = "He.cl";
	std::string sourceStr;
	status = convertToString(filename, sourceStr);
	const char * source = sourceStr.c_str();
	size_t sourceSize[] = {strlen(source)};

	cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);
	status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if(status != 0)
    {
        printf("clBuild failed:%d\n", status);
        char tbuf[0x10000];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0x10000, tbuf, NULL);
        printf("\n%s\n", tbuf);
        return -1;
    }
    cl_kernel kernel = clCreateKernel(program, "HISTEQUA", NULL);
    cl_int clnum = IMG_SIZE;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &clbufR);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &clbufG);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &clbufB);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*) &clMinCdfValR);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*) &clMinCdfValG);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*) &clMinCdfValB);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*) &clCdfR);
    clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*) &clCdfG);
    clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*) &clCdfB);

    struct  timespec nt0, nt1;
    clock_gettime(CLOCK_MONOTONIC,&nt0);

    cl_event ev;
    size_t global_work_size[] = {IMG_W,IMG_H};

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, 0, 0, NULL, &ev);
    clFinish(queue);

    clock_gettime(CLOCK_MONOTONIC,&nt1);
    double allTimeN;
    allTimeN=(nt1.tv_sec+(double)nt1.tv_nsec/1000000000.0)-(nt0.tv_sec+(double)nt0.tv_nsec/1000000000.0);
    printf("  GPU Time:%10.5f sec\n",allTimeN);

    cl_uchar *ptr_r, *ptr_g, *ptr_b;
    ptr_r = (cl_uchar *) clEnqueueMapBuffer(queue, clbufR, CL_TRUE, CL_MAP_READ, 0, IMG_SIZE*sizeof(cl_uchar), 0, NULL, NULL, NULL);
    ptr_g = (cl_uchar *) clEnqueueMapBuffer(queue, clbufG, CL_TRUE, CL_MAP_READ, 0, IMG_SIZE*sizeof(cl_uchar), 0, NULL, NULL, NULL);
    ptr_b = (cl_uchar *) clEnqueueMapBuffer(queue, clbufB, CL_TRUE, CL_MAP_READ, 0, IMG_SIZE*sizeof(cl_uchar), 0, NULL, NULL, NULL);

    SSaveIMGX(ptr_r, ptr_g, ptr_b,"Output_GPU.BMP");

    clReleaseMemObject(clbufR);
    clReleaseMemObject(clbufG);
    clReleaseMemObject(clbufB);
    clReleaseMemObject(clMinCdfValR);
    clReleaseMemObject(clMinCdfValG);
    clReleaseMemObject(clMinCdfValB);
    clReleaseMemObject(clCdfR);
    clReleaseMemObject(clCdfG);
    clReleaseMemObject(clCdfB);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}

int IMG_TransOMP(IMGObj *IObj)
{
	int x,y,index=0, off=0;

    for(y = 0; y < IMG_H; y++)
    {
        for(x = 0; x < IMG_W; x++)
        {
            IObj->sR[off]	 = IObj->InputSrc[index+2];
            IObj->sG[off]  = IObj->InputSrc[index+1];
            IObj->sB[off]  = IObj->InputSrc[index+0];
            off++;
            index += 3;
        }
    }

    int hist_R[256] = {0};  int hist_G[256] = {0};  int hist_B[256] = {0};
    for(y=1; y!=IMG_H; y++){
        index = y*IMG_W;
        for(x=1; x!=IMG_W; x++){
            hist_R[IObj->sR[index]]++;
            hist_G[IObj->sG[index]]++;
            hist_B[IObj->sB[index]]++;
            index++;
        }
    }

    int cdf_R[256] = {0}; int cdf_G[256] = {0}; int cdf_B[256] = {0};
    int v,k;
    for(v=0; v<256; v++){

        if(hist_R[v]>0){
            for(k=v; k<256; k++) cdf_R[k]+=hist_R[v];
        }
        if(hist_G[v]>0){
            for(k=v; k<256; k++) cdf_G[k]+=hist_G[v];
        }
        if(hist_B[v]>0){
            for(k=v; k<256; k++) cdf_B[k]+=hist_B[v];
        }
        //cout << v << ": " << hist_G[v] << endl;
    }

    int minCdfValue_R = 2048*2048; int minCdfValue_G = 2048*2048; int minCdfValue_B = 2048*2048;
    for(int v=0; v<256; v++){
        //cout << v << ": " << cdf_G[v] << endl;
        if((cdf_R[v] != 0) && (cdf_R[v] < minCdfValue_R)) minCdfValue_R = cdf_R[v];
        if((cdf_G[v] != 0) && (cdf_G[v] < minCdfValue_G)) minCdfValue_G = cdf_G[v];
        if((cdf_B[v] != 0) && (cdf_B[v] < minCdfValue_B)) minCdfValue_B = cdf_B[v];
    }

    int sum;
    struct timespec nt0, nt1;
    clock_gettime(CLOCK_MONOTONIC,&nt0);
    #pragma omp parallel num_threads(maxThread)
    {
    	#pragma omp for schedule(dynamic,20) private(y,x,index,sum)
	    for(y=1; y<IMG_H; y++){
	        index = y*IMG_W;
	        for(x=1; x<IMG_W; x++){

	            sum = round(cdf_R[IObj->sR[index]] - minCdfValue_R) / (2048*2048 - minCdfValue_R) * (255);
	            sum = ((sum>255) ? 255 : ((sum<0) ? 0 : sum));
	            IObj->sR[index] = (unsigned)sum;

	            sum = round(cdf_G[IObj->sG[index]] - minCdfValue_G) / (2048*2048 - minCdfValue_G) * (255);
	            sum = ((sum>255) ? 255 : ((sum<0) ? 0 : sum));
	            IObj->sG[index] = (unsigned)sum;

	            sum = round(cdf_B[IObj->sB[index]] - minCdfValue_B) / (2048*2048 - minCdfValue_B) * (255);
	            sum = ((sum>255) ? 255 : ((sum<0) ? 0 : sum));
	            IObj->sB[index] = (unsigned)sum;
	            index++;
	        }
	    }
	}
    clock_gettime(CLOCK_MONOTONIC,&nt1);

    double allTimeN;
    allTimeN=(nt1.tv_sec+(double)nt1.tv_nsec/1000000000.0)-(nt0.tv_sec+(double)nt0.tv_nsec/1000000000.0);
    printf("  OMP Time:%10.5f sec\n",allTimeN);
	return 0;
}

int IMG_Trans(IMGObj *IObj)
{
	int x,y,index=0, off=0;

	for(y = 0; y != IMG_H; y++)
    {
       for(x = 0; x != IMG_W; x++)
        {
		  IObj->sR[off]	 = IObj->InputSrc[index+2];
		  IObj->sG[off]  = IObj->InputSrc[index+1];
		  IObj->sB[off]  = IObj->InputSrc[index+0];
		  off++;
		  index += 3;
        }
    }

    int hist_R[256] = {0};  int hist_G[256] = {0};  int hist_B[256] = {0};
    for(y=1; y!=IMG_H; y++){
        index = y*IMG_W;
        for(x=1; x!=IMG_W; x++){
            hist_R[IObj->sR[index]]++;
            hist_G[IObj->sG[index]]++;
            hist_B[IObj->sB[index]]++;
            index++;
        }
    }

    int cdf_R[256] = {0}; int cdf_G[256] = {0}; int cdf_B[256] = {0};
    for(int v=0; v<256; v++){

        if(hist_R[v]>0){
            for(int k=v; k<256; k++) cdf_R[k]+=hist_R[v];
        }
        if(hist_G[v]>0){
            for(int k=v; k<256; k++) cdf_G[k]+=hist_G[v];
        }
        if(hist_B[v]>0){
            for(int k=v; k<256; k++) cdf_B[k]+=hist_B[v];
        }
    }

    int minCdfValue_R = 2048*2048; int minCdfValue_G = 2048*2048; int minCdfValue_B = 2048*2048;
    for(int v=0; v<256; v++){
        if((cdf_R[v] != 0) && (cdf_R[v] < minCdfValue_R)) minCdfValue_R = cdf_R[v];
        if((cdf_G[v] != 0) && (cdf_G[v] < minCdfValue_G)) minCdfValue_G = cdf_G[v];
        if((cdf_B[v] != 0) && (cdf_B[v] < minCdfValue_B)) minCdfValue_B = cdf_B[v];
    }

    int sum;
    struct timespec nt0, nt1;
    clock_gettime(CLOCK_MONOTONIC,&nt0);
    for(y=1; y!=IMG_H; y++){
        index = y*IMG_W;
        for(x=1; x!=IMG_W; x++){

            sum = round(cdf_R[IObj->sR[index]] - minCdfValue_R) / (2048*2048 - minCdfValue_R) * (255);
            sum = ((sum>255) ? 255 : ((sum<0) ? 0 : sum));
            IObj->sR[index] = (unsigned)sum;

            sum = round(cdf_G[IObj->sG[index]] - minCdfValue_G) / (2048*2048 - minCdfValue_G) * (255);
            sum = ((sum>255) ? 255 : ((sum<0) ? 0 : sum));
            IObj->sG[index] = (unsigned)sum;

            sum = round(cdf_B[IObj->sB[index]] - minCdfValue_B) / (2048*2048 - minCdfValue_B) * (255);
            sum = ((sum>255) ? 255 : ((sum<0) ? 0 : sum));
            IObj->sB[index] = (unsigned)sum;
            index++;
        }
    }
    clock_gettime(CLOCK_MONOTONIC,&nt1);

    double allTimeN;
    allTimeN=(nt1.tv_sec+(double)nt1.tv_nsec/1000000000.0)-(nt0.tv_sec+(double)nt0.tv_nsec/1000000000.0);
    printf("  Normal Time:%10.5f sec\n",allTimeN);
	return 0;
}

int RReadIMG(IMGObj *IObj, const char *fname) {
	FILE             *fp_s = NULL;                        								// source / target file handler
	unsigned int     x,y, width, height, offset;         				  				// for loop counter
	unsigned char    *image_s = NULL;                      				  				// source / target image array
	unsigned short   file_size, rgb_raw_data_offset, bit_per_pixel, byte_per_pixel;  	// bit per pixel, byte_per_pixel
	fp_s = fopen(fname, "rb");

	if (fp_s == NULL)
		printf("fopen fp_s error @ read image\n");

	fseek(fp_s, 10, SEEK_SET);		fread(&rgb_raw_data_offset, sizeof(unsigned int), 1, fp_s);
	fseek(fp_s, 18, SEEK_SET);		fread(&width,  sizeof(unsigned int), 1, fp_s);
	fread(&height, sizeof(unsigned int), 1, fp_s);
	fseek(fp_s, 28, SEEK_SET);      fread(&bit_per_pixel, sizeof(unsigned short), 1, fp_s);
	byte_per_pixel = bit_per_pixel / 8;
    fseek(fp_s, 54, SEEK_SET);
	fread(IObj->InputSrc, sizeof(unsigned char), (size_t)(long)width * height * byte_per_pixel, fp_s);
	if (IObj->InputData == NULL)
		printf("malloc images_s error [RReadIMG]\n");

	fclose(fp_s);
	return 0;
}

int SSaveIMG(IMGObj *IObj, const char *fname) {
   FILE                *fp_t = NULL;                        // target file handler
   unsigned char       *image_t = NULL;                     // target image array
   unsigned int        file_size;                           // file size
   int                 x, y, width = 2048,height = 2048, index;
   image_t = (unsigned char *)malloc((size_t) (width) * height * 3);
   if (image_t == NULL) {   printf("saveIMG malloc image_t error\n");      return -1;     }
   //===========================================================================
   index = 0;
   for(y = 0; y != height; y++)   {      // scal_B[2560][1920];
  		for(x = 0; x != width; x++) {
			*(image_t + 3*index + 2) = (unsigned char)  IObj->sR[index];
       		*(image_t + 3*index + 1) = (unsigned char)  IObj->sG[index];
       		*(image_t + 3*index + 0) = (unsigned char)  IObj->sB[index];
       		index++;
        }
   }
   //===========================================================================
   // write to new bmp
   fp_t = fopen(fname, "wb");
   if (fp_t == NULL) {    printf("saveIMG fopen fname error\n");       return -1;     }
   file_size = width * height * 3 + 54; //16 3 54
   header[2] = (unsigned char)(file_size & 0x000000ff);
   header[3] = (file_size >> 8)  & 0x000000ff;    header[4] = (file_size >> 16) & 0x000000ff;
   header[5] = (file_size >> 24) & 0x000000ff;
   header[18] = width & 0x000000ff;          header[19] = (width >> 8)  & 0x000000ff;
   header[20] = (width >> 16) & 0x000000ff;  header[21] = (width >> 24) & 0x000000ff;
   header[22] = height &0x000000ff;          header[23] = (height >> 8)  & 0x000000ff;
   header[24] = (height >> 16) & 0x000000ff; header[25] = (height >> 24) & 0x000000ff;
   header[28] = 24;
   fwrite(header, sizeof(unsigned char), 54, fp_t);
   fwrite(image_t, sizeof(unsigned char), (size_t)(long)width * height * 3, fp_t);
   fclose(fp_t);
   free(image_t);
   return 0;
}

int SSaveIMGX(unsigned char *ptr_r, unsigned char *ptr_g, unsigned char *ptr_b, const char *fname) {
   FILE                *fp_t = NULL;                        // target file handler
   unsigned char       *image_t = NULL;                     // target image array
   unsigned int        file_size;                           // file size
   int                 x, y, width = 2048,height = 2048, index;
   image_t = (unsigned char *)malloc((size_t) (width) * height * 3);
   if (image_t == NULL) {   printf("saveIMG malloc image_t error\n");      return -1;     }
   //===========================================================================
   index = 0;
   for(y = 0; y != height; y++)   {      // scal_B[2560][1920];
  		for(x = 0; x != width; x++) {
			*(image_t + 3*index + 2) = (unsigned char)  ptr_r[index];
       		*(image_t + 3*index + 1) = (unsigned char)  ptr_g[index];
       		*(image_t + 3*index + 0) = (unsigned char)  ptr_b[index];
       		index++;
        }
   }
   //===========================================================================
   // write to new bmp
   fp_t = fopen(fname, "wb");
   if (fp_t == NULL) {    printf("saveIMG fopen fname error\n");       return -1;     }
   file_size = width * height * 3 + 54; //16 3 54
   header[2] = (unsigned char)(file_size & 0x000000ff);
   header[3] = (file_size >> 8)  & 0x000000ff;    header[4] = (file_size >> 16) & 0x000000ff;
   header[5] = (file_size >> 24) & 0x000000ff;
   header[18] = width & 0x000000ff;          header[19] = (width >> 8)  & 0x000000ff;
   header[20] = (width >> 16) & 0x000000ff;  header[21] = (width >> 24) & 0x000000ff;
   header[22] = height &0x000000ff;          header[23] = (height >> 8)  & 0x000000ff;
   header[24] = (height >> 16) & 0x000000ff; header[25] = (height >> 24) & 0x000000ff;
   header[28] = 24;
   fwrite(header, sizeof(unsigned char), 54, fp_t);
   fwrite(image_t, sizeof(unsigned char), (size_t)(long)width * height * 3, fp_t);
   fclose(fp_t);
   free(image_t);

   return 0;
}


