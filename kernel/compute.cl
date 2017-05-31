__kernel void vie_base(__global unsigned *in, __global unsigned *out){
  int y = get_global_id (1);
  int x = get_global_id (0);

  int couleur = 0;

  if(y == 0 && x == 0){
    couleur = (in[(y+1) * DIM + x] != 0)
      + (in[(y+1) * DIM + (x+1)] != 0)
      + (in[y * DIM + (x+1)] != 0);
  }

  else if(y == 0 && x == DIM-1){
    couleur = (in[(y+1) * DIM + (x-1)] != 0)
      + (in[(y+1) * DIM + x] != 0)
      + (in[y * DIM + (x-1)] != 0);
  }

  else if(x == 0 && y == DIM-1){
    couleur = (in[(y-1) * DIM +x] != 0)
      + (in[(y-1) * DIM + (x+1)] != 0)
      + (in[y * DIM + (x+1)] != 0);
  }

  else if (x == DIM-1 && y == DIM-1){
    couleur = (in[(y-1) * DIM + (x-1)] != 0)
      + (in[(y-1) * DIM +x] != 0)
      + (in[y * DIM + (x-1)] != 0);
  }

  else if(x == 0){
    couleur = (in[(y-1) * DIM +x] != 0)
      + (in[(y-1) * DIM + (x+1)] != 0)
      + (in[(y+1) * DIM + x] != 0)
      + (in[(y+1) * DIM + (x+1)] != 0)
      + (in[y * DIM + (x+1)] != 0);
  }

  else if(y == 0){
    couleur = (in[(y+1) * DIM + (x-1)] != 0)
      + (in[(y+1) * DIM + x] != 0)
      + (in[(y+1) * DIM + (x+1)] != 0)
      + (in[y * DIM + (x-1)] != 0)
      + (in[y * DIM + (x+1)] != 0);
  }

  else if(x == DIM-1){
    couleur = (in[(y-1) * DIM + (x-1)] != 0)
      + (in[(y-1) * DIM +x] != 0)
      + (in[(y+1) * DIM + (x-1)] != 0)
      + (in[(y+1) * DIM + x] != 0)
      + (in[y * DIM + (x-1)] != 0);
  }

  else if(y == DIM-1){
    couleur = (in[(y-1) * DIM + (x-1)] != 0)
      + (in[(y-1) * DIM +x] != 0)
      + (in[(y-1) * DIM + (x+1)] != 0)
      + (in[y * DIM + (x-1)] != 0)
      + (in[y * DIM + (x+1)] != 0);
  }

  else {
    
    couleur = (in[(y-1) * DIM + (x-1)] != 0)
      + (in[(y-1) * DIM +x] != 0)
      + (in[(y-1) * DIM + (x+1)] != 0)
      + (in[(y+1) * DIM + (x-1)] != 0)
      + (in[(y+1) * DIM + x] != 0)
      + (in[(y+1) * DIM + (x+1)] != 0)
      + (in[y * DIM + (x-1)] != 0)
      + (in[y * DIM + (x+1)] != 0);
  }

  if(in[y * DIM + x] != 0)
    if(couleur == 2 || couleur == 3)
      out[y * DIM + x] = 0xFFFF00FF;
    else
      out[y * DIM + x] = 0;
  else
    if(couleur == 3)
      out[y * DIM + x] = 0xFFFF00FF;
    else
      out[y * DIM + x] = 0;
}

__kernel void vie_tuile(__global unsigned *in, __global unsigned *out){

  int x = get_global_id(0);
  int y = get_global_id(1);

  __local unsigned tile[TILEY][TILEX];

  int xloc = get_local_id(0);
  int yloc = get_local_id(1);


  tile[yloc][xloc] = in[y * DIM + x];

  barrier(CLK_LOCAL_MEM_FENCE);

  int couleur;

    if (yloc == 0 && xloc == 0)
      //Mettre les autres condition pour x et y 
      tile[yloc-1][xloc-1] = in[(y - 1) * DIM + (x-1)];
    
    else if(yloc == 0 && xloc == TILEX -1)
      tile[yloc-1][xloc+1] =  in[(y - 1) * DIM + (x+1)];
    
    else if (yloc == 0) 
      tile[yloc-1][xloc] =  in[(y - 1) * DIM + x];

    else if (xloc == 0 && yloc == TILEY - 1)
      tile[yloc+1][xloc-1] = in[(y+1) * DIM + (x-1)];

    else if (xloc == 0)
      tile[yloc][xloc-1] = in[y * DIM + (x-1)];

    else if(xloc == TILEX-1 && yloc == TILEY-1 )
        tile[yloc+1][xloc + 1] = in[(y+1) * DIM + (x+1)];

     barrier(CLK_LOCAL_MEM_FENCE);

  if(y == 0 && x == 0){
    couleur = (tile[yloc+1][xloc] != 0)
      + (tile[yloc+1][xloc+1] != 0)
      + (tile[yloc][xloc+1] != 0);
  }

  else if(y == 0 && x == DIM-1){
    couleur = (tile[yloc+1][xloc-1] != 0)
      + (tile[yloc+1][xloc] != 0)
      + (tile[yloc][xloc-1] != 0);
  }

  else if(x == 0 && y == DIM-1){
    couleur =  (tile[yloc-1][xloc] != 0)
      + (tile[yloc-1][xloc+1] != 0)
      + (tile[yloc][xloc+1] != 0);
  }

  else if (x == DIM-1 && y == DIM-1){
    couleur = (tile[yloc-1][xloc-1] != 0)
      + (tile[yloc-1][xloc] != 0)
      + (tile[yloc][xloc-1] != 0);
  }

  else if(x == 0){
    couleur = (tile[yloc-1][xloc] != 0)
      + (tile[yloc-1][xloc+1] != 0)
      + (tile[yloc+1][xloc] != 0)
      + (tile[yloc+1][xloc+1] != 0)
      + (tile[yloc][xloc+1] != 0);
  }

  else if(y == 0){
    couleur = (tile[yloc+1][xloc-1] != 0)
      + (tile[yloc+1][xloc] != 0)
      + (tile[yloc+1][xloc+1] != 0)
      + (tile[yloc][xloc-1] != 0)
      + (tile[yloc][xloc+1] != 0);
  }

  else if(x == DIM-1){
    couleur = (tile[yloc-1][xloc-1] != 0)
      + (tile[yloc-1][xloc] != 0)
      + (tile[yloc+1][xloc-1] != 0)
      + (tile[yloc+1][xloc] != 0)
      + (tile[yloc][xloc-1] != 0);
  }

  else if(y == DIM-1){
    couleur = (tile[yloc-1][xloc-1] != 0)
      + (tile[yloc-1][xloc] != 0)
      + (tile[yloc-1][xloc+1] != 0)
      + (tile[yloc][xloc-1] != 0)
      + (tile[yloc][xloc+1] != 0);
  }

  else {
    couleur=(tile[yloc-1][xloc-1] != 0)
      + (tile[yloc-1][xloc] != 0)
      + (tile[yloc-1][xloc+1] != 0)
      + (tile[yloc+1][xloc-1] != 0)
      + (tile[yloc+1][xloc] != 0)
      + (tile[yloc+1][xloc+1] != 0)
      + (tile[yloc][xloc-1] != 0)
      + (tile[yloc][xloc+1] != 0);
  }
  

    barrier(CLK_LOCAL_MEM_FENCE);

    if(tile[yloc][xloc] != 0)
      if(couleur == 2 || couleur == 3)
        out[y * DIM + x] = 0xFFFF00FF;
      else
        out[y * DIM + x] = 0;
    else
      if(couleur == 3)
        out[y * DIM + x] = 0xFFFF00FF;
      else
        out[y * DIM + x] = 0;

}


__kernel void transpose_naif (__global unsigned *in, __global unsigned *out)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  out [x * DIM + y] = in [y * DIM + x];
}



__kernel void transpose (__global unsigned *in, __global unsigned *out)
{
  __local unsigned tile [TILEX][TILEY+1];
  int x = get_global_id (0);
  int y = get_global_id (1);
  int xloc = get_local_id (0);
  int yloc = get_local_id (1);

  tile [xloc][yloc] = in [y * DIM + x];

  barrier (CLK_LOCAL_MEM_FENCE);

  out [(x - xloc + yloc) * DIM + y - yloc + xloc] = tile [yloc][xloc];
}



// NE PAS MODIFIER
static unsigned color_mean (unsigned c1, unsigned c2)
{
  uchar4 c;

  c.x = ((unsigned)(((uchar4 *) &c1)->x) + (unsigned)(((uchar4 *) &c2)->x)) / 2;
  c.y = ((unsigned)(((uchar4 *) &c1)->y) + (unsigned)(((uchar4 *) &c2)->y)) / 2;
  c.z = ((unsigned)(((uchar4 *) &c1)->z) + (unsigned)(((uchar4 *) &c2)->z)) / 2;
  c.w = ((unsigned)(((uchar4 *) &c1)->w) + (unsigned)(((uchar4 *) &c2)->w)) / 2;

  return (unsigned) c;
}

// NE PAS MODIFIER
static int4 color_to_int4 (unsigned c)
{
  uchar4 ci = *(uchar4 *) &c;
  return convert_int4 (ci);
}

// NE PAS MODIFIER
static unsigned int4_to_color (int4 i)
{
  return (unsigned) convert_uchar4 (i);
}



// NE PAS MODIFIER
static float4 color_scatter (unsigned c)
{
  uchar4 ci;

  ci.s0123 = (*((uchar4 *) &c)).s3210;
  return convert_float4 (ci) / (float4) 255;
}

// NE PAS MODIFIER: ce noyau est appelé lorsqu'une mise à jour de la
// texture de l'image affichée est requise
__kernel void update_texture (__global unsigned *cur, __write_only image2d_t tex)
{
  int y = get_global_id (1);
  int x = get_global_id (0);
  int2 pos = (int2)(x, y);
  unsigned c;

  c = cur [y * DIM + x];

  write_imagef (tex, pos, color_scatter (c));
}
