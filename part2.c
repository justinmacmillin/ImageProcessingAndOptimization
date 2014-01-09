#include <emmintrin.h>
#include <omp.h>
#include <nmmintrin.h>
#include <stdlib.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
#define NUM_THREADS 8 //number of parallel threads to run at a time
int conv2D(float* in, float* out, int data_size_X, int data_size_Y, float* kernel) {
    
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;

    //size of array with padding
    int padded_x = data_size_X + (2 * kern_cent_X);
    int padded_y = data_size_Y + (2 * kern_cent_Y);
    int padded_size = padded_x*padded_y;
    
    //amount x is unrolled by
    int unroll_size = 16;

    //where the tail loop starts iteration
    int tail_size = (data_size_X/unroll_size)*unroll_size;
    int kern_size = KERNX*KERNY;

    //make copy of array & fill add padding of 0's
    float zedd[1] = {0.0};
    float padded_a[padded_size];
    #pragma omp parallel for
    for (int k = 0; k < padded_size; k++){
        padded_a[k] = 0;
    }

    int copy_unroll = 16;
    #pragma omp parallel for
    for (int k = 0; k < data_size_Y; k++){
        //copying a row from in to the middle data_size_X of the padded_a
        for (int l = 0; l < tail_size; l += copy_unroll){
            __m128 source = _mm_loadu_ps(in + l + (k * data_size_X));
            _mm_storeu_ps(padded_a + l + (k + kern_cent_Y)*padded_x + kern_cent_X, source);
            source = _mm_loadu_ps(in + 4 + l + (k * data_size_X));
            _mm_storeu_ps(padded_a + 4 + l + (k + kern_cent_Y)*padded_x + kern_cent_X, source);
            source = _mm_loadu_ps(in + 8 + l + (k * data_size_X));
            _mm_storeu_ps(padded_a + 8 + l + (k + kern_cent_Y)*padded_x + kern_cent_X, source);
            source = _mm_loadu_ps(in + 12 + l + (k * data_size_X));
            _mm_storeu_ps(padded_a + 12 + l + (k + kern_cent_Y)*padded_x + kern_cent_X, source);
        }
        for (int l = tail_size; l < data_size_X; l++){
            padded_a[l + (k + kern_cent_Y)*padded_x + kern_cent_X] = in[l + (k * data_size_X)];
        }
    }

    //copy kernel into new variable to load whole kernel into cache
    float new_kernel[sizeof(float)*kern_size];
    #pragma omp parallel for 
    for(int k = 0; k < kern_size; k++){
        *(new_kernel+k) = kernel[k];
    }

    // main convolution loop
    omp_set_num_threads(NUM_THREADS);
    int i, j, x, y, kernel_point; float *padded_position, *out_position; __m128 out_vector0, out_vector1, out_vector2, out_vector3, padded_vector, kernel_vector;
    #pragma omp parallel for         
	for (int y = 0; y < data_size_Y; y++){ // the y coordinate of theoutput location we're focusing on
	    for(int x = 0; x < tail_size; x += unroll_size){ // the x coordinate of the output location we're focusing on
		float *out_position = out + x + y * data_size_X;
		__m128 out_vector0 = _mm_loadu_ps(out + x + y * data_size_X);
		__m128 out_vector1 = _mm_loadu_ps(out + 4 + x + y * data_size_X);
		__m128 out_vector2 = _mm_loadu_ps(out + 8 + x + y * data_size_X);
		__m128 out_vector3 = _mm_loadu_ps(out + 12 + x + y * data_size_X);
		for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
		    for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
			float *padded_position = padded_a + (x + i + kern_cent_X) + ((y + kern_cent_Y + j) * (padded_x));
			int kernel_point = (kern_cent_X-i) + (kern_cent_Y-j) * KERNX;
			
			__m128 kernel_vector = _mm_load1_ps(new_kernel+kernel_point);
			__m128 padded_vector = _mm_loadu_ps(padded_position);
			out_vector0 = _mm_add_ps(out_vector0, _mm_mul_ps(padded_vector, kernel_vector));

			padded_vector = _mm_loadu_ps(padded_position+4);
			out_vector1 = _mm_add_ps(out_vector1, _mm_mul_ps(padded_vector, kernel_vector));
			
			padded_vector = _mm_loadu_ps(padded_position+8);
			out_vector2 = _mm_add_ps(out_vector2, _mm_mul_ps(padded_vector, kernel_vector));
			
			padded_vector = _mm_loadu_ps(padded_position+12);
			out_vector3 = _mm_add_ps(out_vector3, _mm_mul_ps(padded_vector, kernel_vector));
		    }
		}
		_mm_storeu_ps(out_position, out_vector0);
		_mm_storeu_ps(out_position+4, out_vector1);
		_mm_storeu_ps(out_position+8, out_vector2);
		_mm_storeu_ps(out_position+12, out_vector3);
	    }
	    for(int x = tail_size; x < data_size_X; x++){
		float *out_position = out + x + y * data_size_X;
		*out_position = 0;
		for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
		    for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
			float *padded_position = padded_a + (x + i + kern_cent_X) + ((y + kern_cent_Y + j) * (padded_x));
			int kernel_point = (kern_cent_X-i) + (kern_cent_Y-j) * KERNX;
			*out_position += *padded_position * new_kernel[kernel_point];
		    }
		}
	    }
	}
    return 1;
}
