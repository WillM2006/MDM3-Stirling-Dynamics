#include "CFDwavemaker.h"
extern "C" void cfd_batch_surface(int nt,const double* times,int np,const double* xy,double* output){
    for(int it=0;it<nt;++it)for(int p=0;p<np;++p)
        output[it*np+p]=wave_SurfElev(xy[2*p],xy[2*p+1],times[it]);
}
extern "C" void cfd_batch_pressure(int nt,const double* times,int np,const double* xyz,double* output){
    for(int it=0;it<nt;++it)for(int p=0;p<np;++p)
        output[it*np+p]=wave_DynPres(xyz[3*p],xyz[3*p+1],xyz[3*p+2],times[it]);
}
