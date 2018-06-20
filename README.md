### POWER-OpenCV: Open Source Computer Vision Library optimized for IBM Power architecture

#### Resources

* Homepage: <http://opencv.org>
* Docs: <http://docs.opencv.org/master/>
* Q&A forum: <http://answers.opencv.org>
* Issue tracking: <https://github.com/opencv/opencv/issues https://github.com/IBM/opencv-power/issues>
* The base version is opencv 3.3.0 

#### Contributing

Please read before starting work on a pull request: <https://github.com/opencv/opencv/wiki/How_to_contribute>
Build and other information, see: <https://github.com/IBM/opencv-power/wiki>

Summary of guidelines:

* One pull request per issue;
* Choose the right base branch;
* Include tests and documentation;
* Clean up "oops" commits before submitting;
* Follow the coding style guide.


#### Compile and install
cd opencv
mkdir build
cd build
cmake -DWITH_JPEG=ON -DWITH_OPENCL=OFF \
-DWITH_OPENMP=ON -DWITH_PTHREADS_PF=OFF \
-DCMAKE_C_FLAGS="-mcpu=power9 -mtune=power9" -DCMAKE_CXX_FLAGS="-mcpu=power9 -mtune=power9" \
-DCMAKE_VERBOSE_MAKEFILE=ON \
-DCMAKE_C_COMPILER=/opt/at11.0/bin/gcc -DCMAKE_CXX_COMPILER=/opt/at11.0/bin/g++ \
..

make -j20
make install

* You could specify the compiler as your default GCC/G++
* It is highly recommend to turn OpenMP on for performance consideration, and it is better to set the environment  OMP_PROC_BIND=true and proper OMP_NUM_THREADS for affinity.
* If you are compiling the code on a POWER8 machine, please replace the cmake c/c++ flags as power8


