using System;
using Binarysharp.Benchmark;
using MathNet.Numerics;
using MathNet.Numerics.Providers.LinearAlgebra;
using MathNet.Numerics.Providers.LinearAlgebra.Mkl;
using MathNet.Numerics.Threading;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Storage;
using System.Runtime.InteropServices;

namespace Performance.LinearAlgebra
{
    public class DenseVectorAdd
    {
        readonly int _rounds;
        readonly Vector<double> _a;
        readonly Vector<double> _b;

        readonly ILinearAlgebraProvider _managed = new ManagedLinearAlgebraProvider();
#if NATIVE
        readonly ILinearAlgebraProvider _mkl = new MklLinearAlgebraProvider();
        readonly ILinearAlgebraProvider _native = new NativeProvider();
        readonly ILinearAlgebraProvider _native_vec = new NativeProvider_Vector();
        readonly ILinearAlgebraProvider _native_par = new NativeProvider_Parallel();
        readonly ILinearAlgebraProvider _native_omp = new NativeProvider_OpenMP();
        readonly ILinearAlgebraProvider _native_opt = new NativeProvider_Optimized();
        readonly ILinearAlgebraProvider _native_opt2 = new NativeProvider_Optimized2();
        readonly ILinearAlgebraProvider _native_amp = new NativeProvider_AMP();
#endif

        public DenseVectorAdd(int size, int rounds)
        {
            _rounds = rounds;

            _b = Vector<double>.Build.Random(size);
            _a = Vector<double>.Build.Random(size);

            _managed.InitializeVerify();
            Control.LinearAlgebraProvider = _managed;

#if NATIVE
            _mkl.InitializeVerify();
            SafeNativePerformanceMethods.set_max_threads(Control.MaxDegreeOfParallelism);
#endif
        }

        public static void Verify(int size)
        {
            var x = new DenseVectorAdd(size, 1);
            var managedResult = x.ManagedProvider();
            var mklResult = x.MklProvider();
            var nativeResult = x.NativeProvider();
            var nativePinnedResult = x.NativeProvider_Pinned();
            var nativeAMPResult = x.NativeProvider_AMP();
            var nativeOMPResult = x.NativeProvider_OpenMP();
            var nativeVectorResult = x.NativeProvider_Vector();
            var nativeParallelResult = x.NativeProvider_Parallel();

            Console.WriteLine(managedResult.ToString());

            if (!managedResult.AlmostEqual(mklResult, 1e-12))
            {
                throw new Exception("MklProvider");
            }
            if (!managedResult.AlmostEqual(nativeResult, 1e-12))
            {
                throw new Exception("NativeProvider");
            }
            if (!managedResult.AlmostEqual(nativePinnedResult, 1e-12))
            {
                throw new Exception("NativeProvider_Pinned");
            }
            if (!managedResult.AlmostEqual(nativeAMPResult, 1e-12))
            {
                throw new Exception("NativeProvider_AMP");
            }
            if (!managedResult.AlmostEqual(nativeOMPResult, 1e-12))
            {
                throw new Exception("NativeProvider_OpenMP");
            }
            if (!managedResult.AlmostEqual(nativeVectorResult, 1e-12))
            {
                throw new Exception("NativeProvider_Vector");
            }
            if (!managedResult.AlmostEqual(nativeParallelResult, 1e-12))
            {
                throw new Exception("NativeProvider_Parallel");
            }
        }

        [BenchSharkTask("AddOperator")]
        public Vector<double> AddOperator()
        {
            var z = _b;
            for (int i = 0; i < _rounds; i++)
            {
                z = _a + z;
            }
            return z;
        }

        [BenchSharkTask("Map2")]
        public Vector<double> Map2()
        {
            var z = _b;
            for (int i = 0; i < _rounds; i++)
            {
                z = _a.Map2((u, v) => u + v, z);
            }
            return z;
        }

        [BenchSharkTask("Loop")]
        public Vector<double> Loop()
        {
            var z = _b;
            for (int i = 0; i < _rounds; i++)
            {
                var aa = ((DenseVectorStorage<double>)_a.Storage).Data;
                var az = ((DenseVectorStorage<double>)z.Storage).Data;
                var ar = new Double[aa.Length];
                for (int k = 0; k < ar.Length; k++)
                {
                    ar[k] = aa[k] + az[k];
                }
                z = Vector<double>.Build.Dense(ar);
            }
            return z;
        }

        [BenchSharkTask("ParallelLoop4096")]
        public Vector<double> ParallelLoop4096()
        {
            var z = _b;
            for (int i = 0; i < _rounds; i++)
            {
                var aa = ((DenseVectorStorage<double>)_a.Storage).Data;
                var az = ((DenseVectorStorage<double>)z.Storage).Data;
                var ar = new Double[aa.Length];
                CommonParallel.For(0, ar.Length, 4096, (u, v) =>
                {
                    for (int k = u; k < v; k++)
                    {
                        ar[k] = aa[k] + az[k];
                    }
                });
                z = Vector<double>.Build.Dense(ar);
            }
            return z;
        }

        [BenchSharkTask("ParallelLoop32768")]
        public Vector<double> ParallelLoop32768()
        {
            var z = _b;
            for (int i = 0; i < _rounds; i++)
            {
                var aa = ((DenseVectorStorage<double>)_a.Storage).Data;
                var az = ((DenseVectorStorage<double>)z.Storage).Data;
                var ar = new Double[aa.Length];
                CommonParallel.For(0, ar.Length, 32768, (u, v) =>
                {
                    for (int k = u; k < v; k++)
                    {
                        ar[k] = aa[k] + az[k];
                    }
                });
                z = Vector<double>.Build.Dense(ar);
            }
            return z;
        }

        [BenchSharkTask("ManagedProvider")]
        public Vector<double> ManagedProvider()
        {
            var z = _b;
            for (int i = 0; i < _rounds; i++)
            {
                var aa = ((DenseVectorStorage<double>)_a.Storage).Data;
                var az = ((DenseVectorStorage<double>)z.Storage).Data;
                var ar = new Double[aa.Length];
                _managed.AddArrays(aa, az, ar);
                z = Vector<double>.Build.Dense(ar);
            }
            return z;
        }

#if NATIVE
        [BenchSharkTask("MklProvider")]
        public Vector<double> MklProvider()
        {
            var z = _b;
            for (int i = 0; i < _rounds; i++)
            {
                var aa = ((DenseVectorStorage<double>)_a.Storage).Data;
                var az = ((DenseVectorStorage<double>)z.Storage).Data;
                var ar = new Double[aa.Length];
                _mkl.AddArrays(aa, az, ar);
                z = Vector<double>.Build.Dense(ar);
            }
            return z;
        }

        [BenchSharkTask("NativeProvider")]
        public Vector<double> NativeProvider()
        {
            var z = _b;
            for (int i = 0; i < _rounds; i++)
            {
                var aa = ((DenseVectorStorage<double>)_a.Storage).Data;
                var az = ((DenseVectorStorage<double>)z.Storage).Data;
                var ar = new Double[aa.Length];
                _native.AddArrays(aa, az, ar);
                z = Vector<double>.Build.Dense(ar);
            }
            return z;
        }

        [BenchSharkTask("NativeProvider_Pinned")]
        public Vector<double> NativeProvider_Pinned()
        {
            var z = _b;
            int n = _a.Count;
            var agch = GCHandle.Alloc(((DenseVectorStorage<double>)_a.Storage).Data, GCHandleType.Pinned);
            var zgch = GCHandle.Alloc(((DenseVectorStorage<double>)z.Storage).Data, GCHandleType.Pinned);

            for (int i = 0; i < _rounds; i++)
            {
                var ar = new Double[n];
                var rgch = GCHandle.Alloc(ar, GCHandleType.Pinned);
                _native.AddArrays(((DenseVectorStorage<double>)_a.Storage).Data, ((DenseVectorStorage<double>)z.Storage).Data, ar);
                zgch.Free();
                zgch = rgch;
                z = Vector<double>.Build.Dense(ar);
            }
            agch.Free();
            zgch.Free();
            return z;
        }

        [BenchSharkTask("NativeProvider_Vector")]
        public Vector<double> NativeProvider_Vector()
        {
            var z = _b;
            for (int i = 0; i < _rounds; i++)
            {
                var aa = ((DenseVectorStorage<double>)_a.Storage).Data;
                var az = ((DenseVectorStorage<double>)z.Storage).Data;
                var ar = new Double[aa.Length];
                _native_vec.AddArrays(aa, az, ar);
                z = Vector<double>.Build.Dense(ar);
            }
            return z;
        }

        [BenchSharkTask("NativeProvider_Parallel")]
        public Vector<double> NativeProvider_Parallel()
        {
            var z = _b;
            for (int i = 0; i < _rounds; i++)
            {
                var aa = ((DenseVectorStorage<double>)_a.Storage).Data;
                var az = ((DenseVectorStorage<double>)z.Storage).Data;
                var ar = new Double[aa.Length];
                _native_par.AddArrays(aa, az, ar);
                z = Vector<double>.Build.Dense(ar);
            }
            return z;
        }

        [BenchSharkTask("NativeProvider_OpenMP")]
        public Vector<double> NativeProvider_OpenMP()
        {
            var z = _b;
            for (int i = 0; i < _rounds; i++)
            {
                var aa = ((DenseVectorStorage<double>)_a.Storage).Data;
                var az = ((DenseVectorStorage<double>)z.Storage).Data;
                var ar = new Double[aa.Length];
                _native_omp.AddArrays(aa, az, ar);
                z = Vector<double>.Build.Dense(ar);
            }
            return z;
        }

        [BenchSharkTask("NativeProvider_Optimized")]
        public Vector<double> NativeProvider_Optimized()
        {
            var z = _b;
            for (int i = 0; i < _rounds; i++)
            {
                var aa = ((DenseVectorStorage<double>)_a.Storage).Data;
                var az = ((DenseVectorStorage<double>)z.Storage).Data;
                var ar = new Double[aa.Length];
                _native_opt.AddArrays(aa, az, ar);
                z = Vector<double>.Build.Dense(ar);
            }
            return z;
        }

        [BenchSharkTask("NativeProvider_Optimized2")]
        public Vector<double> NativeProvider_Optimized2()
        {
            var z = _b;
            for (int i = 0; i < _rounds; i++)
            {
                var aa = ((DenseVectorStorage<double>)_a.Storage).Data;
                var az = ((DenseVectorStorage<double>)z.Storage).Data;
                var ar = new Double[aa.Length];
                _native_opt2.AddArrays(aa, az, ar);
                z = Vector<double>.Build.Dense(ar);
            }
            return z;
        }

        //[BenchSharkTask("NativeProvider_AMP")]
        public Vector<double> NativeProvider_AMP()
        {
            var z = _b;
            for (int i = 0; i < _rounds; i++)
            {
                var aa = ((DenseVectorStorage<double>)_a.Storage).Data;
                var az = ((DenseVectorStorage<double>)z.Storage).Data;
                var ar = new Double[aa.Length];
                _native_amp.AddArrays(aa, az, ar);
                z = Vector<double>.Build.Dense(ar);
            }
            return z;
        }
#endif
    }

#if NATIVE
    public class NativeProvider : ManagedLinearAlgebraProvider
    {
        public override void AddArrays(double[] x, double[] y, double[] result)
        {
            if (y == null)
            {
                throw new ArgumentNullException("y");
            }

            if (x == null)
            {
                throw new ArgumentNullException("x");
            }

            if (x.Length != y.Length)
            {
                throw new ArgumentException();
            }

            if (x.Length != result.Length)
            {
                throw new ArgumentException();
            }

            SafeNativePerformanceMethods.d_vector_add(x.Length, x, y, result);
        }
    }

    public class NativeProvider_Vector : ManagedLinearAlgebraProvider
    {
        public override void AddArrays(double[] x, double[] y, double[] result)
        {
            if (y == null)
            {
                throw new ArgumentNullException("y");
            }

            if (x == null)
            {
                throw new ArgumentNullException("x");
            }

            if (x.Length != y.Length)
            {
                throw new ArgumentException();
            }

            if (x.Length != result.Length)
            {
                throw new ArgumentException();
            }

            SafeNativePerformanceMethods.d_vector_add_vec(x.Length, x, y, result);
        }
    }

    public class NativeProvider_Parallel : ManagedLinearAlgebraProvider
    {
        public override void AddArrays(double[] x, double[] y, double[] result)
        {
            if (y == null)
            {
                throw new ArgumentNullException("y");
            }

            if (x == null)
            {
                throw new ArgumentNullException("x");
            }

            if (x.Length != y.Length)
            {
                throw new ArgumentException();
            }

            if (x.Length != result.Length)
            {
                throw new ArgumentException();
            }

            SafeNativePerformanceMethods.d_vector_add_par(x.Length, x, y, result);
        }
    }

    public class NativeProvider_OpenMP : ManagedLinearAlgebraProvider
    {
        public override void AddArrays(double[] x, double[] y, double[] result)
        {
            if (y == null)
            {
                throw new ArgumentNullException("y");
            }

            if (x == null)
            {
                throw new ArgumentNullException("x");
            }

            if (x.Length != y.Length)
            {
                throw new ArgumentException();
            }

            if (x.Length != result.Length)
            {
                throw new ArgumentException();
            }

            SafeNativePerformanceMethods.d_vector_add_omp(x.Length, x, y, result);
        }
    }

    public class NativeProvider_Optimized : ManagedLinearAlgebraProvider
    {
        public override void AddArrays(double[] x, double[] y, double[] result)
        {
            if (y == null)
            {
                throw new ArgumentNullException("y");
            }

            if (x == null)
            {
                throw new ArgumentNullException("x");
            }

            if (x.Length != y.Length)
            {
                throw new ArgumentException();
            }

            if (x.Length != result.Length)
            {
                throw new ArgumentException();
            }

            SafeNativePerformanceMethods.d_vector_add_opt(x.Length, x, y, result);
        }
    }

    public class NativeProvider_Optimized2 : ManagedLinearAlgebraProvider
    {
        public override void AddArrays(double[] x, double[] y, double[] result)
        {
            if (y == null)
            {
                throw new ArgumentNullException("y");
            }

            if (x == null)
            {
                throw new ArgumentNullException("x");
            }

            if (x.Length != y.Length)
            {
                throw new ArgumentException();
            }

            if (x.Length != result.Length)
            {
                throw new ArgumentException();
            }

            if (x.Length <= (Control.MaxDegreeOfParallelism * 1000))
                SafeNativePerformanceMethods.d_vector_add_vec(x.Length, x, y, result);
            else
                SafeNativePerformanceMethods.d_vector_add_par(x.Length, x, y, result);
        }
    }

    public class NativeProvider_AMP : ManagedLinearAlgebraProvider
    {
        public override void AddArrays(double[] x, double[] y, double[] result)
        {
            if (y == null)
            {
                throw new ArgumentNullException("y");
            }

            if (x == null)
            {
                throw new ArgumentNullException("x");
            }

            if (x.Length != y.Length)
            {
                throw new ArgumentException();
            }

            if (x.Length != result.Length)
            {
                throw new ArgumentException();
            }

            SafeNativePerformanceMethods.d_vector_add_amp(x.Length, x, y, result);
        }
    }
#endif
}
