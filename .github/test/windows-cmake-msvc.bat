@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd %1
cmake -G "Ninja" -S . -B build-msvc-%2 -DCMAKE_BUILD_TYPE=%2 -DCMAKE_C_COMPILER=cl  -DCMAKE_CXX_COMPILER=cl
cmake --build build-msvc-%2 -j
