ImageProcessingAndOptimization
==============================

Brief project description: I was given a slow 2d convolution algorithm and asked to speed it up. In part1.c my partner and I used register blocking, cache blocking, and SSE Intrinsics to speed up the algorithm. In part2.c we made a few changes and implemented openMP to make the algorithm even faster than in part1. 



CS 61C Fall 2013 Project 3: Image Processing and Optimization
TA: Sung Roa Yoon, Ajay Tripathi
Updates
The benchmark had slight bug with image production and produced the wrong colors! If you want to see the actual image, you should download the newer image! (The performance part is not affected)
Background
In the spirit of education and entertainment, we have created, for the first time, an image processing project! This project may look daunting, and it very well may be a difficult project. Hopefully, though, by the end of this though, you guys have much better understanding of vectors, caches, multi-threading, compiler tricks, and image processing.

What you will be doing is trying to optimize a formula for doing 2D convolution on images. 2D convolution is the main machinery behind most forms of image processing. From computer vision to feature detection, from processing MRI scans to making a selfie look like pencil sketches, 2D convolution is the main computation that is involved in a large portion of image processing algorithms. So it is greatly desired to have a very fast implementation of this computationally intensive operation, since image procesing relies heavily upon it. Now the question is why 2D convolution? What is it?

Essentially, 2D convolution is taking a small matrix, called the kernel, and sweeping it across an image, which is a 2D matrix of values that correspond to what is in each pixel. Please watch the following short video for a quick and dirty intro to 2D convolution.

Quick and dirty convolution tutorial

Now that you've watched that video, try to calculate some of the output entries of the following 2D convolution by hand (remember that the kernel is to be flipped over the x and y axes!):



We've provided the basic code for convolution to you as a starting point for your optimization. While it IS possible to finish this project without truly understanding how the naive code works, if you want to get as much optimization as possible, we very highly recommend you try to understand the code!

Even with the naive code that we have given you, you can mess with it and make images change! For example, after running make bench-naive, try:

./bench-naive image2.bmp

Now try opening the two images, image2.bmp, and out_image.bmp!

On the left is the original image. On the right is out_img.bmp that was made in your directory. By default, the initial kernel is set to edge detection, so this new image produces a picture that draws out the edges seen in this picture.



If the image looks different, you probably have the older version of the benchmark! Please update to get correct images! (Performance was not affected)

Pretty cool, eh? :3

One interesting (but not necessary) tidbit on images! Each pixel is encoded as 3 bytes, with the first byte corresponding to the intensity of red, the second the intensity of green, and the last the intensity of blue. These are called RGB values. One little factoid is that Oocoo, the human bird thing in The Legend of Zelda: Twilight Princess, has a not-that-well-known easter egg associated with it. 00CC00 in hex was the RGB colour code of link's hat in the original Legend of Zelda. The more you know! :3

Architecture
What follows is information concerning the computers you will be working on. Some of it will be useful for specific parts of the project, but other elements are there to give you a real world example of the concepts you have been learning, so don't feel forced to use all of the information provided.

You will be developing on Dell Precision T5500 Workstation. They are packing not one, but two Intel Xeon E5620 microprocessors (codename Westmere). Each has 4 cores, for a total of 8 processors, and each core runs at a clock rate of 2.40 GHz.

All caches deal with block sizes of 64 bytes. Each core has an L1 instruction and L1 data cache, both 32 Kibibytes. A core also has a unified L2 cache (same cache for instructions and data) of 256 Kibibytes. The 4 cores on a microprocessor share an L3 cache of 12 Mibibytes. The L1, L2 caches are 8-way associative, while L3 is 16-way associative.

Getting Started
For this project, you will be required to work in groups of two. You are not allowed to show your code to other students. You are to write and debug your own code in groups of 2. Looking at solutions from previous semesters is also strictly prohibited.

To begin, git pull from the directory ~cs61c/proj/03 to a suitable location in your home directory. The following files are provided:

Makefile: allows you to generate the required files using make
benchmark.c: runs a basic performance and correctness test using a randomly generated sequence that simulate a picture. You can also copy a 32-bit color image or 8-bit grayscale image into your directory, and put it as the input to the executables, and you will get the output of what your convolution is actually doing to the picture!
naive.c: a sample implementation for 2D convolution that's simple and straightforward, but inefficient.
You can compile the sample code by running make bench-naive in the project directory. You do not need to modify or submit any of the provided files. Instead, you will be submitting two new files called part1.c (for part 1) and part2.c (for part 2).

You are not permitted to use the following optimizations for either part of the project:

Aligned Loads / Stores (you are allowed to use mm_loadu, but not mm_load)
Optimization based on a specific kernel, as we will be testing your code with variety of kernels.
Optimization by using a DCT (Discrete Cosine Transform) based algorithm or any other algorithm that's runtime is of a different order
Please post all questions about this assignment to Piazza, or ask about them during office hours. Extra office hours will be held for the project.

To obtain the proj3 files, pull from the git repo at ~cs61c/proj/03. For example:

$  mkdir ~/proj3 
$  cd ~/proj3
$  git init
$  git pull ~cs61c/proj/03 master
Functionalities of the benchmark
If you want to test your part2 code, just replace the part1's below with part2!

As you may have noticed by now, if you put an image of the correct, limited format (sorry about that) into the directory and run your function with the image, you can see the image actually modified by the kernel! If you go into benchmark.c, you can actually modify the kernel so that you can see different things if you choose to do so.

make bench-part1; ./bench-part1 IMAGE_NAME.bmp

You can also modify the range of the values the benchmark will test by modifying it in benchmark.c . (Right below the kernel) If you just run the benchmark without any inputs, it will test those range of values and give you an average GFlop/s as well as each individual ones.

make bench-part1; ./bench-part1

If you want to test for a specific size matrix, you can run your program with the input x dimension and input y dimension. That will run the specific dimension a few times and give you the average.

make bench-part1; ./bench-part1 240 240

Part 1: Due Sunday, October 27th, 2013 at 11:59PM (50pt)
Optimize a single image convolution for image of size 240 by 240. Place your solution in a file called part1.c . You'll find a naive implementation in it already for you to start optimizing on.

To compile your improved program, type:

make bench-part1

Tune your solution for image size 240 by 240. To receive full credit, your program needs to achieve a performance level of 9 Gflop/s for images of this size. This requirement is worth 35pt. See the table below for details.

Your code must be able to handle images of size other than 240 by 240 in a reasonably efficient manner. Your code will be tested on images of range 64 to 400 per side, and you will lose points if it fails to achieve an average of 6 Gflop/s on such inputs. This requirement is worth 15pt.

For this part, you are not permitted to use OpenMP directives. Remember that aligned load optimizations are prohibited for both parts.

Submit using:

submit proj3-1.

We recommend that you implement your optimizations in the following order:

Consider Register Blocking (load data into a register once and then use it several times)
Keep the main line (or few lines) of multiplying and adding in your 2D convolution as uncluttered as possible. You don't want if-statements and such slowing down that main area of computation
Optimize loop ordering (see lab 7)
Use SSE Instructions (see lab 8)
Implement Loop Unrolling (see lab 8)
Compiler Tricks (minor modifications to your source code can cause the compiler to produce a faster program)
Make sure to test kernels containing different elements and ranges of dimensions different from the default!

You may also want to consider other tricks we have showed you, such as cache blocking (lab 7), or zero-padding your input.

240 by 240 image convolution

Gflop/s	3.0	4.0	5.0	6.0	6.5	7.0	7.5	8.0	8.5	9.0
Point Value	1	4	7	11	16	21	26	29	32	35
Intermediate Glop/s point values are linearly interpolated based on the point values of the two determined neighbors and then rounded to nearest integer. Note that heavy emphasis is placed on the middle Gflop/s, with the last couple being worth relatively few points. So don't sweat it if you are having trouble squeezing those last few floating point operations out of the machine!

Part 2: Due Sunday, Novermber 3rd, 2013 at 11:59PM (50pt)
The Objective

Now that you have proved your valor on the eternal battlefield that is optimization, it is time to test your mettle on a new beast of a problem. The stakes have increased but so has your arsenal. Now, you have a much higher optimization requirement to fulfill, but you also have OpenMP available for your use. That means you will need to parallelize your computation across multiple cores in order to meet the required goals.

Make sure to test kernels containing different elements and ranges of dimensions different from the default!

Submission and Grading

Place your solution in a file called part2.c. Test it using:

make bench-part2

Submit using:

submit proj3-2.

For Part 2, we will grade your average performance over a range of images whose sizes go from 400 to 1200 both for the height and width. We will try strange image sizes, such of powers-of-2 and primes...so be prepared. Any valid solution that achieves an average performance of 35Gflop/s without using prohibited optimizations (aligned loads and stores, dependency on a specific kernel) will receive full credit (50pt).

Matrix Range Grading

Average Gflop/s	9.0	12.0	16.0	20.0	25.0	30.0	35.0
Point Value	5	10	15	25	35	45	50
Optimization Details
What follows are some useful details for Part 1 and Part 2 of this project.

SSE Instructions

Your code will need to use SSE instructions to get good performance on the processors you are using. Your code should definitely use the _mm_add_ps, _mm_mul_ps, _mm_loadu_ps intrinsics as well as some subset of:

_mm_load1_ps, _mm_storeu_ps, _mm_store_ss _mm_shuffle_ps, _mm_hadd_ps

Depending on how you choose to arrange data in registers and structure your computation, you may not need to use all of these intrinsics (such as _mm_hadd_ps or _mm_shuffle_ps). There are multiple ways to implement image convolution using SSE instructions that perform acceptably well. You probably don't want to use some of the newer SSE instructions, such as those that calculate dot products (though you are welcome to try!). You will need to figure out how to handle images for which side length isn't divisible by 4 (the SSE register width in 32-bit floats), either by using fringes cases (which will probably need optimizing) or by padding the images.

To use the SSE instruction sets supported by the architecture we are on ( MMX, SSE, SSE2, SSE3, SSE4.1, SSE4.2 ) you need to include the header <nmmintrin.h> in your program.

Register Blocking

Your code should re-use values once they have been loaded into registers (both MMX and regular) as much as possible. By reusing values loaded from the image in multiple calculations, you can reduce the total number of times each value is loaded from memory in the course of computing all the entries in C. To ensure that a value gets loaded into a register and reused instead of being loaded from memory repeatedly, you should assign it to a local variable and refer to it using that variable.

Loop Unrolling

Unroll the inner loop of your code to improve your utilization of the pipelined SSE multiplier and adder units, and also reduce the overhead of address calculation and loop condition checking. You can reduce the amount of address calculation necessary in the inner loop of your code by accessing memory using constant offsets, whenever possible.

Padding Matrices

As mentioned previously, it is generally a good idea to pad your images to avoid having to deal with the fringe cases. This implies copying the entire image into a re-formatted buffer with no fringes (e.g. copy 3x4 matrice into 4x4 buffer). Be sure to fill the padded elements with zeros.

Cache Blocking

To optimize cache performance so performance doesn't drop off for larger image sizes, you should load chunks of the image and do all the necessary work on them before moving on to the next chunks. The cache sizes listed in the Architecture section should come in handy here.

OpenMP

Once you have code that performs well for large images, it will be time to wield the formidable power offered to you by the machine's 8 cores. Use at least one OpenMP pragma statement to parallelize your computations.

Miscellaneous Tips

You may also wish to try copying small blocks of your image into contiguous chunks of memory order to improve spatial locality.

References
Chellappa, S., Franchetti, F., and Püschel, M. 2008.How To Write Fast Numerical Code: A Small Introduction,Lecture Notes in Computer Science 5235, 196–259.
Lam, M. S., Rothberg, E. E, and Wolf, M. E. 1991.The Cache Performance and Optimization of Blocked Algorithms,ASPLOS'91, 63–74.
Intel Instrinsics Guide (see Lab 8)
Intel® 64 and IA-32 Architectures Optimization Reference Manual
OpenMP reference sheet
OpenMP Reference Page from LLNL
