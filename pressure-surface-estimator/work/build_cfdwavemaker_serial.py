"""Build the independent library locally; no system packages are modified."""
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1];repo=ROOT/'work/CFDwavemaker'
source=repo/'src';build=repo/'builds/macos';build.mkdir(parents=True,exist_ok=True)
file=source/'Irregular.cpp';text=file.read_text()
if '#if defined(_OPENMP)\n#include <omp.h>\n#endif' not in text:
    file.write_text(text.replace('#include <omp.h>','#if defined(_OPENMP)\n#include <omp.h>\n#endif'))
version=(source/'VERSION').read_text().strip()
files=['CFDwavemaker','Stokes5','FentonStream','Irregular','Utils','Wavemaker','probes','input_reader','lsgrid_spline','sigma_fft','seabed']
subprocess.run(['clang++','-O3','-fPIC','-pthread','-std=c++11','-dynamiclib',f'-DCFDWM_VERSION="v{version}"',
                *[str(source/(f+'.cpp')) for f in files],'-o',str(build/'libCFDwavemaker.dylib')],check=True)
subprocess.run(['clang++','-O3','-fPIC','-std=c++11','-dynamiclib',str(ROOT/'work/cfd_batch.cpp'),
                '-I',str(source),'-L',str(build),'-lCFDwavemaker','-o',str(build/'libCFDbatch.dylib')],check=True)
