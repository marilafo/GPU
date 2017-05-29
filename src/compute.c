
#include "compute.h"
#include "graphics.h"
#include "debug.h"
#include "ocl.h"
#include <unistd.h>
#include <omp.h>

#include <stdbool.h>

//KERNEL=scrollup ./prog -v 2 -s 1024 

unsigned version = 0;

void first_touch_v1 (void);
void first_touch_v2 (void);

unsigned compute_v0 (unsigned nb_iter);
unsigned compute_v1 (unsigned nb_iter);
unsigned compute_v2 (unsigned nb_iter);
unsigned compute_v3 (unsigned nb_iter);
unsigned compute_v4 (unsigned nb_iter);
unsigned compute_v5 (unsigned nb_iter);
unsigned compute_v6 (unsigned nb_iter);
unsigned compute_v7 (unsigned nb_iter);
unsigned compute_v8 (unsigned nb_iter);




void_func_t first_touch [] = {
  NULL,
  first_touch_v1,
  first_touch_v2,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
};

int_func_t compute [] = {
  compute_v0,
  compute_v1,
  compute_v2,
  compute_v3,
  compute_v4,
  compute_v5,
  compute_v6,
  compute_v7,
  compute_v8,
};

char *version_name [] = {
  "Séquentielle",
  "OpenMP for simple",
  "OpenMP for sans collapse",
  "Séquentielle avec tuile",
  "OpenMP for avec tuile",
  "Séquentielle optimisé",
  "OpenMP for optimisé",
  "OpenMP task tuiléé",
  "OpenMP task optimisé",
};

unsigned opencl_used [] = {
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
};

///////////////////////////// Version séquentielle simple
void calcul_vie(int i, int j){
	int alive = 0;
	alive = (cur_img(i-1, j-1) !=0)
	  + (cur_img(i-1, j) != 0)
	  + (cur_img(i+1, j-1) != 0)
	  + (cur_img(i, j-1) != 0)
	  + (cur_img(i, j+1) != 0)
	  + (cur_img(i-1, j+1) != 0)
	  + (cur_img(i+1, j) != 0)
	  + (cur_img(i+1, j+1) !=0);

	if(cur_img(i,j) != 0){
	  if ((alive == 2) || (alive == 3))
	    next_img(i,j) = 0xFFFF00FF;
	  else
	    next_img(i,j) = 0;
	}
	else{
	  if(alive == 3)
	    next_img(i,j) = 0xFFFF00FF;
	  else
	    next_img(i,j) = 0;
	}
	
	alive = 0;
}

unsigned compute_v0 (unsigned nb_iter)
{
  for (unsigned test = 1; test <= nb_iter; test++) {
    for (int i = 1; i < DIM-1; i++)
      for (int j = 1; j < DIM-1; j++){
		calcul_vie(i,j);
      } 
    swap_images ();
    }
  return 0;
}


///////////////////////////// Version OpenMP for simple

void first_touch_v1 ()
{
  
}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v1(unsigned nb_iter)
{
	#pragma omp parallel
	{
		#pragma omp for
	  	for (unsigned test = 1; test <= nb_iter; test++) {
		   	#pragma omp parallel for collapse(2)
		   	for (int i = 1; i < DIM-1; i++)
		   		for (int j = 1; j < DIM-1; j++){
					calcul_vie(i,j);
		
	   		}
    	}
    }
    swap_images ();
  return 0; 
}



///////////////////////////// Version OpenMP avec for sans collapse

void first_touch_v2 ()
{

}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v2(unsigned nb_iter)
{
	#pragma omp parallel
	{

		#pragma omp for
	  	for (unsigned test = 1; test <= nb_iter; test++) {
		   	#pragma omp parallel for 
		   	for (int i = 1; i < DIM-1; i++)
		   		#pragma omp parallel for
		   		for (int j = 1; j < DIM-1; j++){
					calcul_vie(i,j);
	   		}
    	}
    }
    swap_images ();
  return 0; 
}

///////////////////////////// Version séquentielle avec tuile
#define GRAIN 16
unsigned tranche = 0;

volatile int cont = 0;

void get_tuile(int *ret, int i, int j){
	ret[0] = (i == 0) ? 1 : i * tranche;
  	ret[1] = (j == 0) ? 1 : j * tranche;
  	ret[2] = (i == GRAIN-1) ? DIM-2 : (i+1) * tranche - 1;
  	ret[3] = (j == GRAIN-1) ? DIM-2 : (j+1) * tranche - 1;

  PRINT_DEBUG ('c', "Descente: bloc(%d,%d) couvre (%d,%d)-(%d,%d)\n", i, j, ret[0], ret[1], ret[2], ret[3]);
  
}


unsigned jeu_vie_seq (int a, int b)
{
   	{

	   	int ret[4];
	   	get_tuile(ret, a, b);

    	for (int i = ret[0]; i <= ret[2]; i++)
		    for (int j = ret[1]; j <= ret[3]; j++){
				calcul_vie(i,j);
		    }
	}
  	return 0;
}


// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v3 (unsigned nb_iter)
{ 
  	tranche = DIM / GRAIN;

  	for (unsigned test = 1; test <= nb_iter; test++) {
  		cont = 0;
  		for (int i=0; i < GRAIN; i++)
    		for (int j=0; j < GRAIN; j++){
      			jeu_vie_seq (i, j);
    		}
	}
   	swap_images ();
  return cont;
}

//////////////////////////version OMP for tuilé
unsigned jeu_vie_v4 (int a, int b)
{

	int ret[4];
   	get_tuile(ret, a, b);
   	
   	#pragma omp parallel
   	{
   		

  		#pragma omp for collapse(2)
    	for (int i = ret[0]; i <= ret[2]; i++)
	    	for (int j = ret[1]; j <= ret[3]; j++){
				calcul_vie(i,j);
			
	    }
	} 
  	return 0;
}


// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v4 (unsigned nb_iter)
{ 
  	tranche = DIM / GRAIN;

  	#pragma omp parallel for
  	for (unsigned test = 1; test <= nb_iter; test++) 
  	{
  		cont = 0;
  		#pragma omp parallel for collapse(2)
  		for (int i=0; i < GRAIN; i++)
    		for (int j=0; j < GRAIN; j++){
      			jeu_vie_v4 (i, j);
    		}
	}
   	swap_images ();

  return cont;
}


//////////////////////////version séquentielle optimisé

volatile int test_matrice; 
volatile int cellule[GRAIN][GRAIN+1];


unsigned jeu_vie_v5 (int a, int b)
{
   	{
   		int ret[4];
   		get_tuile(ret, a, b);

    	for (int i = ret[0]; i <= ret[2]; i++)
		    for (int j = ret[1]; j <= ret[3]; j++){
				calcul_vie(i,j);
			
	    }
	}

	if(test_matrice == 1)
	    cellule[a][b] = 1;
	else
	    cellule[a][b] = 0;
    
  	return 0;
}


// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v5 (unsigned nb_iter)
{ 
  	tranche = DIM / GRAIN;

  	for (unsigned test = 1; test <= nb_iter; test++) {
  		cont = 0;
  		for (unsigned m = 0 ; m < GRAIN ; m++){
  			for(unsigned n = 0; n < GRAIN ; n++){
  				cellule[m][n] = 1;
  			}
  		}

  		for (int i=0; i < GRAIN; i++)
    		for (int j=0; j < GRAIN; j++){
    			if(cellule[i][j] == 0 
	    			&& cellule[i][j-1] == 0 
	    			&& cellule[i][j+1] == 0 
	    			&& cellule[i-1][j] == 0
	    			&& cellule[i+1][j] == 0
	    			&& cellule[i-1][j-1] == 0
	    			&& cellule[i-1][j+1] == 0 
	    			&& cellule[i+1][j-1] == 0
	    			&& cellule[i+1][j+1] == 0 )
	    			continue;
	    		test_matrice = 0;
      			jeu_vie_v5 (i, j);
    		}
	}
   	swap_images ();

  return cont;
}


//////////////////////////version OMP for optimisé

volatile int test_matrice; 
volatile int cellule[GRAIN][GRAIN+1];


unsigned jeu_vie_v6 (int a, int b)
{
	int ret[4];
   	get_tuile(ret, a, b);

	#pragma omp parallel
   	{
   		#pragma omp for collapse(2)
    	for (int i = ret[0]; i <= ret[2]; i++)
		    for (int j = ret[1]; j <= ret[3]; j++){
				calcul_vie(i,j);
			
	    }
	}
    
   	if(test_matrice == 1)
	    cellule[a][b] = 1;
	else
	    cellule[a][b] = 0;

  	return 0;
}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v6 (unsigned nb_iter)
{ 
  	tranche = DIM / GRAIN;
  	#pragma omp parallel
  	{
  	#pragma omp for
  	for (unsigned test = 1; test <= nb_iter; test++) {
  		cont = 0;

  		#pragma omp parallel for collapse(2)
  		for (unsigned m = 0 ; m < GRAIN ; m++){
  			for(unsigned n = 0; n < GRAIN ; n++){
  				#pragma omp critical
  				cellule[m][n] = 1;
  			}
  		}

  		#pragma omp parallel for collapse(2)
  		for (int i=0; i < GRAIN; i++)
    		for (int j=0; j < GRAIN; j++){
    			if(cellule[i][j] == 0 
	    			&& cellule[i][j-1] == 0 
	    			&& cellule[i][j+1] == 0 
	    			&& cellule[i-1][j] == 0
	    			&& cellule[i+1][j] == 0
	    			&& cellule[i-1][j-1] == 0
	    			&& cellule[i-1][j+1] == 0 
	    			&& cellule[i+1][j-1] == 0
	    			&& cellule[i+1][j+1] == 0 )
	    			continue;
	    		test_matrice = 0;
      			jeu_vie_v6 (i, j);
    		}
	}
	}	
   	swap_images ();

  return cont;
}

//////////////////////////version OMP TASK tuilée

unsigned jeu_vie_v7 (int a, int b)
{
	int ret[4];
	get_tuile(ret, a, b);
	
	#pragma omp parallel
   	{

    	for (int i = ret[0]; i <= ret[2]; i++)
		    for (int j = ret[1]; j <= ret[3]; j++){
				calcul_vie(i,j);
		    }
	}
  	return 0;
}


// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v7 (unsigned nb_iter)
{ 
  	tranche = DIM / GRAIN;

  	for (unsigned test = 1; test <= nb_iter; test++) {
  		cont = 0;
  		for (int i=0; i < GRAIN; i++)
    		for (int j=0; j < GRAIN; j++){
      			jeu_vie_seq (i, j);
    		}
	}
   	swap_images ();
  return cont;
}

//////////////////////////version OMP TASK optimisé

unsigned compute_v8 (unsigned nb_iter)
{ 
  	return 2;
}