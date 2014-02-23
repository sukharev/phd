@echo off
"C:\Program Files (x86)\CMake 2.8\bin\cmake.exe" -E make_directory "C:/SVN/Dev/optix_gl/ptx"
if errorlevel 1 goto VCReportError
"C:\Program Files (x86)\CMake 2.8\bin\cmake.exe" -D verbose:BOOL=ON -D build_configuration:STRING=Debug -D "generated_file:STRING=C:/SVN/Dev/optix_gl/ptx/isgShadows_generated_ppm_rtpass.cu.ptx" -D "generated_cubin_file:STRING=C:/SVN/Dev/optix_gl/ptx/isgShadows_generated_ppm_rtpass.cu.ptx.cubin.txt" -P "C:/SVN/Dev/optix_gl/CMakeFiles/isgShadows_generated_ppm_rtpass.cu.ptx.cmake"
if errorlevel 1 goto VCReportError

if errorlevel 1 goto VCReportError
goto VCEnd
:VCReportError
echo Project : error PRJ0019: A tool returned an error code from "Building NVCC ptx file "
exit 1
:VCEnd