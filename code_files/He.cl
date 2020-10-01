__kernel void HISTEQUA (
 __global unsigned char* R, __global unsigned char* G, __global unsigned char* B,
 __global int* minCdfValR, __global int* minCdfValG, __global int* minCdfValB,
 __global int* clCdfR, __global int* clCdfG, __global int* clCdfB)
{
	int col= get_global_id(0);
	int row = get_global_id(1);
	int width = get_global_size(0);
    int height = get_global_size(1);
	
	double sum = 0.0;

	int index = row*height+col;
    sum = (clCdfR[R[index]] - *minCdfValR) / (2048.0*2048.0 - *minCdfValR) * (255.0);
    sum = ((sum>255) ? 255 : ((sum<0) ? 0 : sum));
    R[index] = (unsigned)sum;

    sum = (clCdfG[G[index]] - *minCdfValG) / (2048.0*2048.0 - *minCdfValG) * (255.0);
    sum = ((sum>255) ? 255 : ((sum<0) ? 0 : sum));
    G[index] = (unsigned)sum;

    sum = (clCdfB[B[index]] - *minCdfValB) / (2048.0*2048.0 - *minCdfValB) * (255.0);
    sum = ((sum>255) ? 255 : ((sum<0) ? 0 : sum));
    B[index] = (unsigned)sum;
}



