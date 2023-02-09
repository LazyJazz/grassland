@echo off
cmake -G "Ninja" -B %1 -DCMAKE_BUILD_TYPE=%2 "-DCMAKE_C_COMPILER=C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin/clang.exe" "-DCMAKE_CXX_COMPILER=C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin/clang++.exe" "-DCMAKE_RC_COMPILER=C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin/llvm-rc.exe"
cmake --build %1 -j
