#include <complex>
#include "wrapper_common.h"
#include <amp.h>
using namespace concurrency;

namespace
{
    static int num_threads = 1;
}

template<typename T>
inline void vector_add(const int n, const T x[], const T y[], T result[])
{
    // No vertorization or parallelization
    #pragma loop(no_vector)
    #pragma loop(no_parallel)
    for (int i = 0; i < n; i++)
    {
        result[i] = x[i] + y[i];
    }
}

template<typename T>
inline void vector_add_vec(const int n, const T x[], const T y[], T result[])
{
    // This will be auto vectorized
    #pragma loop(no_parallel)
    #pragma loop(ivdep)
    for (int i = 0; i < n; i++)
    {
        result[i] = x[i] + y[i];
    }
}

template<typename T>
inline void vector_add_omp(const int n, const T x[], const T y[], T result[])
{
    // OpenMP parallel
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        result[i] = x[i] + y[i];
    }
}

template<typename T>
inline void vector_add_par(const int n, const T x[], const T y[], T result[])
{
    // Parallel loop using maximum number of threads
    #pragma loop(hint_parallel(0))
    #pragma loop(ivdep)
    for (int i = 0; i < n; i++)
    {
        result[i] = x[i] + y[i];
    }
}

template<typename T>
inline void vector_add_opt(const int n, const T x[], const T y[], T result[])
{
    if (n <= (::num_threads * 1000))
    {
        // Vectorized
        #pragma loop(no_parallel)
        #pragma loop(ivdep)
        for (int i = 0; i < n; i++)
        {
            result[i] = x[i] + y[i];
        }
    }
    else
    {
        // Parallelized and vectorized using maximum number of threads
        #pragma loop(hint_parallel(0))
        #pragma loop(ivdep)
        for (int i = 0; i < n; i++)
        {
            result[i] = x[i] + y[i];
        }
    }
}

template<typename T>
inline void vector_add_amp(const int n, const T x[], const T y[], T result[])
{
    array_view<T, 1> xv(n, const_cast<T*>(x));
    array_view<T, 1> yv(n, const_cast<T*>(y));
    array_view<T, 1> resultv(n, result);

    parallel_for_each(
        resultv.extent,
        [=](index<1> idx) restrict(amp, cpu)
        {
            resultv[idx] = xv[idx] + yv[idx];
        }
    );

    resultv.synchronize();
}


extern "C" {

    DLLEXPORT void set_max_threads(const int num_threads)
    {
        ::num_threads = num_threads;
    }

    DLLEXPORT void s_vector_add(const int n, const float x[], const float y[], float result[]){
        vector_add(n, x, y, result);
    }

    DLLEXPORT void d_vector_add(const int n, const double x[], const double y[], double result[]){
        vector_add(n, x, y, result);
    }

    DLLEXPORT void c_vector_add(const int n, const std::complex<float> x[], const std::complex<float> y[], std::complex<float> result[]){
        vector_add(n, x, y, result);
    }

    DLLEXPORT void z_vector_add(const int n, const std::complex<double> x[], const std::complex<double> y[], std::complex<double> result[]){
        vector_add(n, x, y, result);
    }

    DLLEXPORT void s_vector_add_vec(const int n, const float x[], const float y[], float result[]){
        vector_add_vec(n, x, y, result);
    }

    DLLEXPORT void d_vector_add_vec(const int n, const double x[], const double y[], double result[]){
        vector_add_vec(n, x, y, result);
    }

    DLLEXPORT void c_vector_add_vec(const int n, const std::complex<float> x[], const std::complex<float> y[], std::complex<float> result[]){
        vector_add_vec(n, x, y, result);
    }

    DLLEXPORT void z_vector_add_vec(const int n, const std::complex<double> x[], const std::complex<double> y[], std::complex<double> result[]){
        vector_add_vec(n, x, y, result);
    }

    DLLEXPORT void s_vector_add_omp(const int n, const float x[], const float y[], float result[]){
        vector_add_omp(n, x, y, result);
    }

    DLLEXPORT void d_vector_add_omp(const int n, const double x[], const double y[], double result[]){
        vector_add_omp(n, x, y, result);
    }

    DLLEXPORT void c_vector_add_omp(const int n, const std::complex<float> x[], const std::complex<float> y[], std::complex<float> result[]){
        vector_add_omp(n, x, y, result);
    }

    DLLEXPORT void z_vector_add_omp(const int n, const std::complex<double> x[], const std::complex<double> y[], std::complex<double> result[]){
        vector_add_omp(n, x, y, result);
    }

    DLLEXPORT void s_vector_add_par(const int n, const float x[], const float y[], float result[]){
        vector_add_par(n, x, y, result);
    }

    DLLEXPORT void d_vector_add_par(const int n, const double x[], const double y[], double result[]){
        vector_add_par(n, x, y, result);
    }

    DLLEXPORT void c_vector_add_par(const int n, const std::complex<float> x[], const std::complex<float> y[], std::complex<float> result[]){
        vector_add_par(n, x, y, result);
    }

    DLLEXPORT void z_vector_add_par(const int n, const std::complex<double> x[], const std::complex<double> y[], std::complex<double> result[]){
        vector_add_par(n, x, y, result);
    }

    DLLEXPORT void s_vector_add_opt(const int n, const float x[], const float y[], float result[]){
        vector_add_opt(n, x, y, result);
    }

    DLLEXPORT void d_vector_add_opt(const int n, const double x[], const double y[], double result[]){
        vector_add_opt(n, x, y, result);
    }

    DLLEXPORT void c_vector_add_opt(const int n, const std::complex<float> x[], const std::complex<float> y[], std::complex<float> result[]){
        vector_add_opt(n, x, y, result);
    }

    DLLEXPORT void z_vector_add_opt(const int n, const std::complex<double> x[], const std::complex<double> y[], std::complex<double> result[]){
        vector_add_opt(n, x, y, result);
    }
    DLLEXPORT void s_vector_add_amp(const int n, const float x[], const float y[], float result[]){
        vector_add_amp(n, x, y, result);
    }

    DLLEXPORT void d_vector_add_amp(const int n, const double x[], const double y[], double result[]){
        vector_add_amp(n, x, y, result);
    }

    //DLLEXPORT void c_vector_add_amp(const int n, const std::complex<float> x[], const std::complex<float> y[], std::complex<float> result[]){
    //    vector_add_amp(n, x, y, result);
    //}

    //DLLEXPORT void z_vector_add_amp(const int n, const std::complex<double> x[], const std::complex<double> y[], std::complex<double> result[]){
    //    vector_add_amp(n, x, y, result);
    //}
}
