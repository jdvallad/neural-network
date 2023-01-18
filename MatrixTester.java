import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

public class MatrixTester {
    public static void main(String[] args) throws Exception {
        DataIterator trainer = new DataIterator(100, "../mnist/training/data.ser");
        while (trainer.hasNextBatch()) {
            int size = 0;
            double pooledMean = -1;
            double pooledStd = -1;
            for (DataPair data : trainer.nextBatch()) {
                Matrix m = data.input;
                if (size == 0) {
                    size = m.getRows() * m.getColumns();
                    pooledMean = getMean(m);
                    pooledStd = sd(m);
                    continue;
                }
                pooledMean = poolMeans(size, pooledMean, m.getRows() * m.getColumns(), getMean(m));
                pooledStd = poolStandardDeviations(size, pooledStd, m.getRows() * m.getColumns(), sd(m));
                size += m.getRows() * m.getColumns();
                System.out.println("Mean: " + pooledMean + ", std: " + pooledStd);
            }
        }
    }

    public static double getMean(Matrix m) {
        List<Double> input = DoubleStream.of(m.getCells()).boxed().collect(Collectors.toList());
        return input.stream()
                .mapToDouble(d -> d)
                .average()
                .orElse(0.0);
    }

    public static double sd(Matrix m) {
        // Step 1:
        List<Double> table = DoubleStream.of(m.getCells()).boxed().collect(Collectors.toList());

        double mean = getMean(m);
        double temp = 0;

        for (int i = 0; i < table.size(); i++) {
            double val = table.get(i);

            // Step 2:
            double squrDiffToMean = Math.pow(val - mean, 2);

            // Step 3:
            temp += squrDiffToMean;
        }

        // Step 4:
        double meanOfDiffs = (double) temp / (double) (table.size());

        // Step 5:
        return Math.sqrt(meanOfDiffs);
    }

    public static double poolMeans(int size1, double mean1, int size2, double mean2) {
        double doubleSize1 = (double) size1;
        double doubleSize2 = (double) size2;
        return (doubleSize1 * mean1 + doubleSize2 * mean2) / (doubleSize1 + doubleSize2);
    }

    public static double poolStandardDeviations(int size1, double std1, int size2, double std2) {
        double doubleSize1 = (double) size1;
        double doubleSize2 = (double) size2;
        return Math.sqrt((((doubleSize1 - 1) * Math.pow(std1, 2)) + ((doubleSize2 - 1) * Math.pow(std2, 2)))
                / (doubleSize1 + doubleSize2 - 2));
    }
}