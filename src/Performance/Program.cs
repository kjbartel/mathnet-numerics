using System;
using System.Linq;
using Binarysharp.Benchmark;
using ConsoleDump;
using MathNet.Numerics.Statistics;

namespace Performance
{
    public class Program
    {
        public static void Main()
        {
            //LinearAlgebra.DenseVectorAdd.Verify(10);
            LinearAlgebra.DenseVectorAdd.Verify(100);
            //LinearAlgebra.DenseVectorAdd.Verify(1000);
            Benchmark(new LinearAlgebra.DenseVectorAdd(10, 1000), 4000, "X-Tiny (10) - 1000x4000 iterations");
            Benchmark(new LinearAlgebra.DenseVectorAdd(16, 1000), 4000, "Tiny (16) - 1000x4000 iterations");
            Benchmark(new LinearAlgebra.DenseVectorAdd(100, 1000), 2000, "X-Small (100) - 1000x2000 iterations");
            Benchmark(new LinearAlgebra.DenseVectorAdd(1000, 1000), 1000, "Small (1'000) - 1000x1000 iterations");
            Benchmark(new LinearAlgebra.DenseVectorAdd(10000, 1000), 100, "Medium (10'000) - 1000x100 iterations");
            Benchmark(new LinearAlgebra.DenseVectorAdd(100000, 100), 100, "Large (100'000) - 100x100 iterations");
            Benchmark(new LinearAlgebra.DenseVectorAdd(10000000, 10), 10, "X-Large (10'000'000) - 10x10 iterations");

            //LinearAlgebra.DenseMatrixProduct.Verify(5);
            LinearAlgebra.DenseMatrixProduct.Verify(100);
            Benchmark(new LinearAlgebra.DenseMatrixProduct(10, 100), 100, "10 - 100x100 iterations");
            Benchmark(new LinearAlgebra.DenseMatrixProduct(25, 100), 100, "25 - 100x100 iterations");
            Benchmark(new LinearAlgebra.DenseMatrixProduct(50, 80), 100, "50 - 100x100 iterations");
            Benchmark(new LinearAlgebra.DenseMatrixProduct(100, 40), 40, "100 - 40x40 iterations");
            Benchmark(new LinearAlgebra.DenseMatrixProduct(250, 10), 20, "250 - 20x10 iterations");
            Benchmark(new LinearAlgebra.DenseMatrixProduct(500, 4), 10, "500 - 10x4 iterations");
            Benchmark(new LinearAlgebra.DenseMatrixProduct(1000, 4), 10, "1000 - 10x4 iterations");
        }

        static void Benchmark(object obj, uint iterations, string suffix = null)
        {
            var bench = new BenchShark(true);
            var result = bench.EvaluateDecoratedTasks(obj, iterations);
            var results = result.FastestEvaluations.Select(x =>
            {
                var series = x.Iterations.Select(it => (double)it.ElapsedTicks).ToArray();
                Array.Sort(series);
                var summary = SortedArrayStatistics.FiveNumberSummary(series);
                var ms = ArrayStatistics.MeanStandardDeviation(series);
                return new { x.Name, Mean = ms.Item1, StdDev = ms.Item2, Min = summary[0], Q1 = summary[1], Median = summary[2], Q3 = summary[3], Max = summary[4] };
            }).ToArray();
            var top = results[0];
            var managed = results.Single(x => x.Name.StartsWith("Managed"));
            var label = string.IsNullOrEmpty(suffix) ? obj.GetType().FullName : string.Concat(obj.GetType().FullName, ": ", suffix);
            results.Select(x => new
            {
                x.Name,
                Mean = Math.Round(x.Mean), StdDev = Math.Round(x.StdDev),
                Min = Math.Round(x.Min), Q1 = Math.Round(x.Q1), Median = Math.Round(x.Median), Q3 = Math.Round(x.Q3), Max = Math.Round(x.Max),
                TopSlowdown = Math.Round(x.Median/top.Median, 2),
                ManagedSpeedup = Math.Round(managed.Median/x.Median, 2)
            }).Dump(label);
        }
    }
}
