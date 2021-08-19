You need to install torch7 and git clone mc-cnn https://github.com/jzbontar/mc-cnn

Install Torch, OpenCV 2.4, and png++.

Run the following commands in the same directory as this README file.

Compile the shared libraries:

$ cp Makefile.proto Makefile
$ make

The command should produce two files: libadcensus.so and libcv.so.

Copy libadcensus.so and adcensus.cu in here
