/**
* Group Info:
* rwsnyde2 Richard W Snyder
* kshanka2 Koushik Shankar
*/

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <time.h>
#include "mpi.h"
#define __DEBUG

#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)

extern int tpdt(double *t, double dt, double end_time);
extern int nproc, rank;
double* getmyrows(double *row,double* u1,int n_points_per_quadrant,int rank);
double* getmycolumns(double *column,double* u1,int n_points_per_quadrant,int rank);
double* getmycorner(double *corner,double* u1,int n_points_per_quadrant,int rank);
void setmyrow(double* row,double **u1,int n_points_per_quadrant,int rank);
void setmycolumn(double* column,double **u1,int n_points_per_quadrant,int rank);
void setmycorner(double* corner,double **u1,int n_points_per_quadrant,int rank);




#define TSCALE 1.0
#define VSQR 0.1

/**************************************
* void __cudaSafeCall(cudaError err, const char *file, const int line)
* void __cudaCheckError(const char *file, const int line)
*
* These routines were taken from the GPU Computing SDK
* (http://developer.nvidia.com/gpu-computing-sdk) include file "cutil.h"
**************************************/
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef __DEBUG

#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
              file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
  } while ( 0 );
#pragma warning( pop )
#endif  // __DEBUG
  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef __DEBUG
#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment if not needed.
    /*err = cudaThreadSynchronize();
    if( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }*/
  } while ( 0 );
#pragma warning( pop )
#endif // __DEBUG
  return;
}

// function to get the rows required for computation
double* getmyrows(double *row,double* u1,int n_points_per_quadrant,int rank)
{
  if (rank == 0)
  {
      int index = 2 + (n_points_per_quadrant + 4) * (n_points_per_quadrant);
      memcpy(row,&u1[index],sizeof(double)*n_points_per_quadrant);
      index = index + n_points_per_quadrant + 4;
      memcpy(&row[n_points_per_quadrant],&u1[index],sizeof(double)*n_points_per_quadrant);

  } 
  if (rank == 1)
  {
    int index = 2 + (n_points_per_quadrant + 4) * (2);
    memcpy(row,&u1[index],sizeof(double)*n_points_per_quadrant);
    index = index + n_points_per_quadrant + 4;
    memcpy(&row[n_points_per_quadrant],&u1[index],sizeof(double)*n_points_per_quadrant);

  }

  if (rank == 2)
  {
      int index = 2 + (n_points_per_quadrant + 4) * (n_points_per_quadrant);
      memcpy(row,&u1[index],sizeof(double)*n_points_per_quadrant);
      index = index + n_points_per_quadrant + 4;
      memcpy(&row[n_points_per_quadrant],&u1[index],sizeof(double)*n_points_per_quadrant);

  }

  if (rank == 3)
  {
    int index = 2 + (n_points_per_quadrant + 4) * (2);
    memcpy(row,&u1[index],sizeof(double)*n_points_per_quadrant);
    index = index + n_points_per_quadrant + 4;
    memcpy(&row[n_points_per_quadrant],&u1[index],sizeof(double)*n_points_per_quadrant);

  }

  return row;

}

// function to get the columns required for computation
double* getmycolumns(double *column,double* u1,int n_points_per_quadrant,int rank)
{

    if(rank == 0)
    {
        int index = n_points_per_quadrant + (n_points_per_quadrant + 4) * (2);
        int count = 0;
        for (int i = 0; i < n_points_per_quadrant; i++)
        {
            memcpy(&column[count],&u1[index],2 * sizeof(double));
            count = count + 2;
            index = index + n_points_per_quadrant + 4;
        }

    }

    if(rank == 1)
    {
        int index = n_points_per_quadrant + (n_points_per_quadrant + 4) * (2);
        int count = 0;
        for (int i = 0; i < n_points_per_quadrant; i++)
        {
            memcpy(&column[count],&u1[index],2 * sizeof(double));
            count = count + 2;
            index = index + n_points_per_quadrant + 4;
        }

    }
    
    if(rank == 2)
    {
        int index = 2 + (n_points_per_quadrant + 4) * (2);
        int count = 0;
        for (int i = 0; i < n_points_per_quadrant; i++)
        {
            memcpy(&column[count],&u1[index],2 * sizeof(double));
            count = count + 2;
            index = index + n_points_per_quadrant + 4;
        }
    }
    if(rank == 3)
    {
        int index = 2 + (n_points_per_quadrant + 4) * (2);
        int count = 0;
        for (int i = 0; i < n_points_per_quadrant; i++)
        {
            memcpy(&column[count],&u1[index],2 * sizeof(double));
            count = count + 2;
            index = index + n_points_per_quadrant + 4;
        }

    }



    return column;

}

// function to get the corner tiles required for computation
double* getmycorner(double* corner,double* u1,int n_points_per_quadrant,int rank)
{

    if(rank == 0)
    {
        int index = (1 + n_points_per_quadrant) + (1 + n_points_per_quadrant) * (n_points_per_quadrant + 4);
        *corner = u1[index];
    }

    if(rank == 1)
    {
        int index = (1 + n_points_per_quadrant)+ (2) * (n_points_per_quadrant + 4);
        *corner = u1[index];
    }

    if(rank == 2)
    {
        int index = 2 + (1 + n_points_per_quadrant) * (n_points_per_quadrant + 4);
        *corner = u1[index];
    }

    if(rank == 3)
    {
        int index = 2 + (2) * (n_points_per_quadrant + 4);
        *corner = u1[index];
    }

    return corner;



}

void setmyrow(double* row,double **u1,int n_points_per_quadrant,int rank)
{
    if (rank == 0)
    {
        int count = 0;
        int index = 2 + (2 + n_points_per_quadrant) * (n_points_per_quadrant + 4);
        memcpy(&((*u1)[index]),row,sizeof(double)*n_points_per_quadrant);
        count = count + n_points_per_quadrant;
        index = index + (n_points_per_quadrant + 4);
        memcpy(&((*u1)[index]),&row[count],sizeof(double)*n_points_per_quadrant);


    }

    if (rank == 1)
    {
        int count = 0;
        int index = 2;
        memcpy(&((*u1)[index]),row,sizeof(double)*n_points_per_quadrant);
        count = count + n_points_per_quadrant;
        index = index + (n_points_per_quadrant + 4);
        memcpy(&((*u1)[index]),&row[count],sizeof(double)*n_points_per_quadrant);       

    }

    if (rank == 2)
    {
        int count = 0;
        int index = 2 + (2 + n_points_per_quadrant) * (n_points_per_quadrant + 4);
        memcpy(&((*u1)[index]),row,sizeof(double)*n_points_per_quadrant);
        count = count + n_points_per_quadrant;
        index = index + (n_points_per_quadrant + 4);
        memcpy(&((*u1)[index]),&row[count],sizeof(double)*n_points_per_quadrant);

    }

    if (rank == 3)
    {
        int count = 0;
        int index = 2;
        memcpy(&((*u1)[index]),row,sizeof(double)*n_points_per_quadrant);
        count = count + n_points_per_quadrant;
        index = index + (n_points_per_quadrant + 4);
        memcpy(&((*u1)[index]),&row[count],sizeof(double)*n_points_per_quadrant);       

    }
}

void setmycolumn(double* column,double **u1,int n_points_per_quadrant,int rank)
{

    if (rank == 0)
    {
        int index = (n_points_per_quadrant + 2) +(n_points_per_quadrant + 4)* 2;
        int count = 0;
        for (int i = 0; i < n_points_per_quadrant; i++)
        {
            memcpy(&((*u1)[index]),&column[count],2 * sizeof(double));
            count = count + 2;
            index = index + n_points_per_quadrant + 4;

        }

    }

    if (rank == 1)
    {
        int index = (n_points_per_quadrant + 2) +(n_points_per_quadrant + 4)* 2;
        int count = 0;
        for (int i = 0; i < n_points_per_quadrant; i++)
        {
            memcpy(&((*u1)[index]),&column[count],2 * sizeof(double));
            count = count + 2;
            index = index + n_points_per_quadrant + 4;

        }


    }
    
    if (rank == 2)
    {
        int index =  (n_points_per_quadrant + 4)* 2;
        int count = 0;
        for (int i = 0; i < n_points_per_quadrant; i++)
        {
            memcpy(&((*u1)[index]),&column[count],2 * sizeof(double));
            count = count + 2;
            index = index + n_points_per_quadrant + 4;

        }
    }

    if (rank == 3)
    {
        int index = (n_points_per_quadrant + 4)* 2;
        int count = 0;
        for (int i = 0; i < n_points_per_quadrant; i++)
        {
            memcpy(&((*u1)[index]),&column[count],2 * sizeof(double));
            count = count + 2;
            index = index + n_points_per_quadrant + 4;

        }

    }



}

void setmycorner(double* corner,double **u1,int n_points_per_quadrant,int rank)
{

    if(rank == 0)
    {
        int index = (2 + n_points_per_quadrant) + (n_points_per_quadrant + 4) * (2 + n_points_per_quadrant);
        memcpy(&((*u1)[index]),corner,sizeof(double));

    }

    if (rank == 1)
    {
        int index = (2 + n_points_per_quadrant) + (n_points_per_quadrant + 4);
        memcpy(&((*u1)[index]),corner,sizeof(double));

    }

    if (rank == 2)
    {
        int index = 2 + (n_points_per_quadrant + 4) * (2 + n_points_per_quadrant);
        memcpy(&((*u1)[index]),corner,sizeof(double));
    }

    if (rank == 3)
    {
        int index = 1 + (n_points_per_quadrant + 4);
        memcpy(&((*u1)[index]),corner,sizeof(double));
    }


}

__global__ void evolve13(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t,int rank)
{
    int dx = threadIdx.x + blockIdx.x * blockDim.x;
    int i = dx / n;
    int j = dx % n;

	int idx = (i+2)*(n + 4) + (j+2);
    int x = n + 4;

    int flag = 0;

    if (rank == 0)
    {
        if ( i <= 1 || j <= 1)
        {
            un[idx] = 0.;
            flag = 1;
        }

    }

    if (rank == 1)
    {
        if ( i >= n - 2 || j <= 1)
        {
            un[idx] = 0.;
            flag = 1;
        }
    }

    if (rank == 2)
    {
        if ( i <= 1 || j >= n - 2)
        {
            un[idx] = 0.;
            flag = 1;
        }
    }

    if (rank == 3)
    {
        if ( i >= n - 2 || j >= n - 2)
        {
            un[idx] = 0.;
            flag = 1;
        }
    }


    if (flag == 0)
    {
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *
        ((uc[idx-1] + uc[idx+1] + uc[idx + x] + uc[idx-x] + // west east north south
        0.25 * (uc[idx - x - 1] + uc[idx - x + 1] + uc[idx + x - 1] + uc[idx + x + 1]) + // northwest northeast southwest southeast
        0.125 * (uc[idx - 2] + uc[idx + 2] + uc[idx - (2*x)] + uc[idx + (2*x)]) - // westwest easteast northnorth southsouth
        5.5 * uc[idx])/(h * h) + (-expf(-TSCALE * t) * pebbles[idx]));

    }

    __syncthreads();
}



void run_gpu(double *u,double * u0,double * u1, 
  double *pebbles,int n,
  double h, double end_time, int nthreads,int rank)

{
    cudaEvent_t kstart, kstop;
    float ktime;
    double t, dt;

    int num_blocks = (n/nthreads)*(n/nthreads);
    int threads_per_block = nthreads * nthreads;
    double *u_d, *u0_d, *u1_d,*pebbles_d;

    double *row,*column,*corner,*recv_row,*recv_column,*recv_corner;
    corner = (double*)malloc(sizeof(double));
    row = (double*)malloc(sizeof(double) * (2*n));
    column = (double*)malloc(sizeof(double) * (2*n));
    recv_corner = (double*)malloc(sizeof(double));
    recv_row = (double*)malloc(sizeof(double) * (2*n));
    recv_column = (double*)malloc(sizeof(double) * (2*n));

    CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
    CUDA_CALL(cudaEventCreate(&kstop));
    
	cudaMalloc((void **) &u_d, sizeof(double) * (n + 4) * (n + 4)); 
	cudaMalloc((void **) &u0_d, sizeof(double) * (n + 4) * (n + 4)); 
	cudaMalloc((void **) &u1_d, sizeof(double) * (n + 4) * (n + 4)); 
	cudaMalloc((void **) &pebbles_d, sizeof(double) * (n + 4) * (n + 4)); 

	// Copy pebbles and u values from Host to Device 
	cudaMemcpy(u0_d, u0, sizeof(double) * (n + 4) * (n + 4), cudaMemcpyHostToDevice);
	cudaMemcpy(pebbles_d, pebbles, sizeof(double) * (n + 4) * (n + 4), cudaMemcpyHostToDevice);
    
    CUDA_CALL(cudaEventRecord(kstart, 0));
    t = 0.;
    dt = h / 2.;


    while(1)
    {

        MPI_Status request_status[6];
        MPI_Request requests[6];


        row = getmyrows(row,u1,n,rank);
        column = getmycolumns(column,u1,n,rank);
        corner = getmycorner(corner,u1,n,rank);

        if (rank == 0)
        {
            MPI_Isend(row, 2*n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD,&requests[0]);	
            MPI_Isend(column, 2*n, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD,&requests[1]);	
            MPI_Isend(corner, 1, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD,&requests[2]);	
        }

        if(rank == 1)
        {
            MPI_Isend(row, 2*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&requests[0]);	
            MPI_Isend(column, 2*n, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD,&requests[1]);	
            MPI_Isend(corner, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD,&requests[2]);	
        }

        if (rank == 2)
        {
            MPI_Isend(row, 2*n, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD,&requests[0]);	
            MPI_Isend(column, 2*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&requests[1]);	
            MPI_Isend(corner, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD,&requests[2]);	
        }

        if (rank == 3)
        {
            MPI_Isend(row, 2*n, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD,&requests[0]);	
            MPI_Isend(column, 2*n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD,&requests[1]);	
            MPI_Isend(corner, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&requests[2]);	
        }


        if (rank == 0)
        {
            MPI_Irecv(recv_row, 2*n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD,&requests[3]);	
            MPI_Irecv(recv_column, 2*n, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD,&requests[4]);	
            MPI_Irecv(recv_corner, 1, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD,&requests[5]);
        }

        if(rank == 1)
        {
            MPI_Irecv(recv_row, 2*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&requests[3]);	
            MPI_Irecv(recv_column, 2*n, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD,&requests[4]);	
            MPI_Irecv(recv_corner, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD,&requests[5]);
        }


        if (rank == 2)
        {
            MPI_Irecv(recv_row, 2*n, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD,&requests[3]);	
            MPI_Irecv(recv_column, 2*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&requests[4]);	
            MPI_Irecv(recv_corner, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD,&requests[5]);
        }

        if (rank == 3)
        {
            MPI_Irecv(recv_row, 2*n, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD,&requests[3]);	
            MPI_Irecv(recv_column, 2*n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD,&requests[4]);	
            MPI_Irecv(recv_corner, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&requests[5]);
        }



        MPI_Waitall(6,requests,request_status);

        setmyrow(recv_row,&u1,n,rank);
        setmycolumn(recv_column,&u1,n,rank);
        setmycorner(recv_corner,&u1,n,rank);

        cudaMemcpy(u1_d, u1, sizeof(double) * (n + 4) * (n + 4), cudaMemcpyHostToDevice);


        evolve13<<< num_blocks,threads_per_block >>>(u_d, u1_d, u0_d, pebbles_d, n, h, dt, t, rank);

		cudaMemcpy(u0_d, u1_d, sizeof(double) * (n + 4) * (n + 4), cudaMemcpyDeviceToDevice);
        cudaMemcpy(u1_d, u_d, sizeof(double) * (n + 4) * (n + 4), cudaMemcpyDeviceToDevice);
        cudaMemcpy(u1,u1_d, sizeof(double) * (n + 4) * (n + 4), cudaMemcpyDeviceToHost);

        if(!tpdt(&t,dt,end_time)) break;

        MPI_Barrier(MPI_COMM_WORLD);


    }




	/* Stop GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
	printf("GPU computation: %f msec\n", ktime);

	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));

    cudaMemcpy(u, u_d, sizeof(double) * (n + 4) * (n + 4), cudaMemcpyDeviceToHost);
	cudaFree(u_d);
	cudaFree(u0_d);
	cudaFree(u1_d);
    cudaFree(pebbles_d);
    free(row);
    free(recv_row);
    free(column);
    free(recv_column);
    free(corner);
    free(recv_corner);
}