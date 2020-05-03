/**
* Group Info:
* rwsnyde2 Richard W Snyder
* kshanka2 Koushik Shankar
*/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>    
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include "mpi.h"

#define _USE_MATH_DEFINES

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

void init(double *u, double *pebbles, int n);
void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
void evolve13pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
int tpdt(double *t, double dt, double end_time);
void print_heatmap_cpu(const char *filename, double *u, int n, double h);
void print_heatmap_gpu(const char *filename, double *u, int n, double h);
void init_pebbles(double *p, int pn, int n);

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time);

extern void run_gpu(double *u_gpu_quadrant,double *u_i0_quadrant,double *u_i1_quadrant, 
  double *pebs_quadrant,int n_points_per_quadrant,
  double h, double end_time, int nthreads, int rank);

void set_init_index(int *start_row,int *start_column,int rank,int n_points_per_proc);

void quad_div(double *pebs,double *u_i0,double *u_i1,double *pebs_quadrant,
    double *u_i0_quadrant,double *u_i1_quadrant,int n_points_per_quadrant,
    int start_row, int start_column);

// divide the grid into 4 quadrants to be passed to the 
// 4 nodes
void set_init_index(int *start_row,int *start_column,int rank,int n_points_per_quadrant)
{
    // top left quadrant
    if (rank == 0)
    {
        *start_row = 0;
        *start_column = 0;
    }

    // bottom left quadrant
    if (rank == 1)
    {
        *start_row = n_points_per_quadrant;
        *start_column = 0;
    }

    // top right quadrant
    if (rank == 2)
    {
        *start_row = 0;
        *start_column = n_points_per_quadrant;
    }

    // bottom right quadrant
    if (rank == 3)
    {
        *start_row = n_points_per_quadrant;
        *start_column = n_points_per_quadrant;
    }
}

void quad_div(double *pebs,double *u_i0,double *u_i1,double *pebs_quadrant,
double *u_i0_quadrant,double *u_i1_quadrant,int n_points_per_quadrant,
int start_row, int start_column)
{
  int i, j;
  for (i = 0; i< n_points_per_quadrant; i++)
  {
      for (j = 0; j < n_points_per_quadrant; j++)
      {
          int index_quadrant = (j+2) + (i+2) * (n_points_per_quadrant + 4);
          int index_global = (j + start_column) + (i + start_row)*(n_points_per_quadrant*2);
          pebs_quadrant[index_quadrant] = pebs[index_global];
          u_i0_quadrant[index_quadrant] = u_i0[index_global];
          u_i1_quadrant[index_quadrant] = u_i1[index_global];
      }
  }
}



int main(int argc, char *argv[])
{

  setbuf(stdout, NULL);

  // check command line argument count
  if(argc != 5)
  {
    printf("Usage: %s npoints npebs time_finish nthreads \n",argv[0]);
    return 0;
  }

	int world_size, rank;

  // do MPI initialization
	MPI_Init(&argc,&argv);

  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	double elapsed_gpu;
	struct timeval gpu_start, gpu_end;
  
  // grab command line arguments
  int     npoints   = atoi(argv[1]);
  int     npebs     = atoi(argv[2]);
  double  end_time  = (double)atof(argv[3]);
  int     nthreads  = atoi(argv[4]);
  int     narea     = npoints * npoints;

  int start_row;
  int start_column;

  // each quadrant will represent an
  // (n/2) * (n/2) grid
  int n_points_per_quadrant = (npoints / 2);
  int n_area_quadrant = (n_points_per_quadrant + 4) * (n_points_per_quadrant + 4);

  double *u_i0, *u_i1,*u_i0_quadrant,*u_i1_quadrant, *pebs_quadrant,*u_gpu_quadrant;
  double *pebs;
  double h;

  h = (XMAX - XMIN)/npoints;


  u_i0_quadrant = (double*)malloc(sizeof(double) * n_area_quadrant);
  u_i1_quadrant = (double*)malloc(sizeof(double) * n_area_quadrant);
  pebs = (double*)malloc(sizeof(double) * narea);
  u_i0 = (double*)malloc(sizeof(double) * narea);
  u_i1 = (double*)malloc(sizeof(double) * narea);
  pebs_quadrant = (double*)malloc(sizeof(double) * n_area_quadrant);

  u_gpu_quadrant = (double*)malloc(sizeof(double) * n_area_quadrant);

  set_init_index(&start_row,&start_column,rank,n_points_per_quadrant);
  if (rank == 0)
  {
      init_pebbles(pebs, npebs, npoints);
  }

  // broadcast arrays to all nodes
  MPI_Bcast(pebs, narea, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
  init(u_i0, pebs, npoints);
  init(u_i1, pebs, npoints);

  quad_div(pebs,u_i0,u_i1,pebs_quadrant,u_i0_quadrant,
    u_i1_quadrant,n_points_per_quadrant,start_row,start_column);

  // run GPU
  gettimeofday(&gpu_start, NULL);

  run_gpu(u_gpu_quadrant, u_i0_quadrant, u_i1_quadrant, 
    pebs_quadrant, n_points_per_quadrant,
      h, end_time, nthreads, rank);

  gettimeofday(&gpu_end, NULL);
  elapsed_gpu = ((gpu_end.tv_sec + gpu_end.tv_usec * 1e-6)-(
                  gpu_start.tv_sec + gpu_start.tv_usec * 1e-6));
  
  printf("GPU took %f seconds\n", elapsed_gpu);



  if(rank ==0) 
  {
    print_heatmap_gpu("lake_f_0.dat", u_gpu_quadrant, n_points_per_quadrant, h);
  }
  else if(rank==1)
  {
    print_heatmap_gpu("lake_f_1.dat", u_gpu_quadrant, n_points_per_quadrant, h);
  }
  else if(rank==2)
  {
    print_heatmap_gpu("lake_f_2.dat", u_gpu_quadrant, n_points_per_quadrant, h);
  }
  else if(rank==3)
  {
    print_heatmap_gpu("lake_f_3.dat", u_gpu_quadrant, n_points_per_quadrant, h);
  }

  // run CPU if node 0
  if (rank == 0)
  {
    double *cpu_i0 = (double*)malloc(sizeof(double) * npoints*npoints);
    double *cpu_i1 = (double*)malloc(sizeof(double) * npoints*npoints);     
    double *u_cpu = (double*)malloc(sizeof(double) * npoints*npoints);
    
    init(cpu_i0, pebs, npoints);
    init(cpu_i1, pebs, npoints);     
    
    struct timeval cpu_start, cpu_end;
    double elapsed_cpu;
    gettimeofday(&cpu_start, NULL);
    run_cpu(u_cpu, cpu_i0, cpu_i1, pebs, npoints, h, end_time);
    gettimeofday(&cpu_end, NULL);

    elapsed_cpu = ((cpu_end.tv_sec + cpu_end.tv_usec * 1e-6)-(cpu_start.tv_sec + cpu_start.tv_usec * 1e-6));
    printf("CPU took %f seconds\n", elapsed_cpu);

    print_heatmap_cpu("lake_f.dat", u_cpu, npoints, h);
  }



  MPI_Barrier(MPI_COMM_WORLD);
  
  // clean up
  free(u_i0);
	free(u_i1);
	free(pebs);
	MPI_Finalize(); 


}

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time)
{
  double *un, *uc, *uo;
  double t, dt;

  un = (double*)malloc(sizeof(double) * n * n);
  uc = (double*)malloc(sizeof(double) * n * n);
  uo = (double*)malloc(sizeof(double) * n * n);

  memcpy(uo, u0, sizeof(double) * n * n);
  memcpy(uc, u1, sizeof(double) * n * n);

  t = 0.;
  dt = h / 2.;

  while(1)
  {
    evolve13pt(un, uc, uo, pebbles, n, h, dt, t);

    memcpy(uo, uc, sizeof(double) * n * n);
    memcpy(uc, un, sizeof(double) * n * n);

    if(!tpdt(&t,dt,end_time)) break;
  }

  memcpy(u, un, sizeof(double) * n * n);
}

void init_pebbles(double *p, int pn, int n)
{
  int i, j, k, idx;
  int sz;

  srand( time(NULL) );
  memset(p, 0, sizeof(double) * n * n);

  for( k = 0; k < pn ; k++ )
  {
    i = rand() % (n - 4) + 2;
    j = rand() % (n - 4) + 2;
    sz = rand() % MAX_PSZ;
    idx = j + i * n;
    p[idx] = (double) sz;
  }
}

double f(double p, double t)
{
  return -expf(-TSCALE * t) * p;
}

int tpdt(double *t, double dt, double tf)
{
  if((*t) + dt > tf) return 0;
  (*t) = (*t) + dt;
  return 1;
}

void init(double *u, double *pebbles, int n)
{
  int i, j, idx;

  for(i = 0; i < n ; i++)
  {
    for(j = 0; j < n ; j++)
    {
      idx = j + i * n;
      u[idx] = f(pebbles[idx], 0.0);
    }
  }
}

void evolve13pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)
{
  int i, j, idx;

  for( i = 0; i < n; i++)
  {
    for( j = 0; j < n; j++)
    {
      idx = j + i * n;

      // check to see that you do not go out of the 
      // array bounds
      if( i <= 1 || i >= n - 2 || j <= 1 || j >= n - 2)
      {
        un[idx] = 0.;
      }
      else
      {
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *
                  ((uc[idx-1] + uc[idx+1] + uc[idx + n] + uc[idx-n] + // west east north south
                  0.25 * (uc[idx - n - 1] + uc[idx - n + 1] + uc[idx + n - 1] + uc[idx + n + 1]) + // northwest northeast southwest southeast
                  0.125 * (uc[idx - 2] + uc[idx + 2] + uc[idx - (2*n)] + uc[idx + (2*n)]) - // westwest easteast northnorth southsouth
                  5.5 * uc[idx])/(h * h) + f(pebbles[idx], t));
      }
    }
  }
}

void print_heatmap_cpu(const char *filename, double *u, int n, double h)
{
  int i, j, idx;

  FILE *fp = fopen(filename, "w");

  for( i = 0; i < n; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      idx = j + i * n;
      fprintf(fp, "%f %f %f\n", i*h, j*h, u[idx]);
    }
  }

  fclose(fp);
}


void print_heatmap_gpu(const char *filename, double *u, int n, double h)
{
  int i, j, idx;

  FILE *fp = fopen(filename, "w");

  for( i = 0; i < n; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      idx = (j + 2) + (i + 2) * (n+4);
      fprintf(fp, "%f %f %f\n", i*h, j*h, u[idx]);
    }
  }

  fclose(fp);
}