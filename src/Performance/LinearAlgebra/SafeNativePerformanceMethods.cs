using MathNet.Numerics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using System.Threading.Tasks;

namespace Performance.LinearAlgebra
{
#if NATIVE
    [SuppressUnmanagedCodeSecurity, SecurityCritical]
    internal static class SafeNativePerformanceMethods
    {
        const string _DllName = "NativePerformance.dll";

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void set_max_threads(int num_threads);

        #region No vectorization or parallelisation

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void s_vector_add(int n, float[] x, float[] y, [In, Out] float[] result);

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void d_vector_add(int n, double[] x, double[] y, [In, Out] double[] result);

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void c_vector_add(int n, Complex32[] x, Complex32[] y, [In, Out] Complex32[] result);

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void z_vector_add(int n, Complex[] x, Complex[] y, [In, Out] Complex[] result);

        #endregion

        #region Auto vectorization used

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void s_vector_add_vec(int n, float[] x, float[] y, [In, Out] float[] result);

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void d_vector_add_vec(int n, double[] x, double[] y, [In, Out] double[] result);

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void c_vector_add_vec(int n, Complex32[] x, Complex32[] y, [In, Out] Complex32[] result);

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void z_vector_add_vec(int n, Complex[] x, Complex[] y, [In, Out] Complex[] result);

        #endregion

        #region Auto parallelization used

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void s_vector_add_par(int n, float[] x, float[] y, [In, Out] float[] result);

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void d_vector_add_par(int n, double[] x, double[] y, [In, Out] double[] result);

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void c_vector_add_par(int n, Complex32[] x, Complex32[] y, [In, Out] Complex32[] result);

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void z_vector_add_par(int n, Complex[] x, Complex[] y, [In, Out] Complex[] result);

        #endregion

        #region Optimized auto vectorization and parallelization used

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void s_vector_add_opt(int n, float[] x, float[] y, [In, Out] float[] result);

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void d_vector_add_opt(int n, double[] x, double[] y, [In, Out] double[] result);

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void c_vector_add_opt(int n, Complex32[] x, Complex32[] y, [In, Out] Complex32[] result);

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void z_vector_add_opt(int n, Complex[] x, Complex[] y, [In, Out] Complex[] result);

        #endregion

        #region OpenMP used

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void s_vector_add_omp(int n, float[] x, float[] y, [In, Out] float[] result);

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void d_vector_add_omp(int n, double[] x, double[] y, [In, Out] double[] result);

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void c_vector_add_omp(int n, Complex32[] x, Complex32[] y, [In, Out] Complex32[] result);

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void z_vector_add_omp(int n, Complex[] x, Complex[] y, [In, Out] Complex[] result);

        #endregion

        #region AMP used

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void s_vector_add_amp(int n, float[] x, float[] y, [In, Out] float[] result);

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void d_vector_add_amp(int n, double[] x, double[] y, [In, Out] double[] result);

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void c_vector_add_amp(int n, Complex32[] x, Complex32[] y, [In, Out] Complex32[] result);

        [DllImport(_DllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void z_vector_add_amp(int n, Complex[] x, Complex[] y, [In, Out] Complex[] result);

        #endregion
    }
#endif
}
