### HWPV DLL Build Instructions

Dependency on https://github.com/nlohmann/json, MIT license

Dependency on https://github.com/icl-utk-edu/blaspp, BSD 3-clause license 

Dependency on https://github.com/icl-utk-edu/lapackpp, BSD 3-clause license

To build the DLL:

    mkdir build
    cd build
    cmake ..
    cmake --build . --config Release
