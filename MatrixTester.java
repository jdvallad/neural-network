import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

public class MatrixTester {
    public static void main(String[] args) throws Exception {
        DataIterator trainer = new DataIterator(100, "../mnist/training/input.ser", "../mnist/training/output.ser");
        DataIterator validator = new DataIterator(100, "../mnist/validation/input.ser",
                "../mnist/validation/output.ser");
        DataIterator tester = new DataIterator(100, "../mnist/testing/input.ser", "../mnist/testing/output.ser");

        List<Double> means = new ArrayList<>();
        List<Double> stds = new ArrayList<>();
        double mean, std;
        double avg = 46.409632999270976;
        double spread = 81.39233883364163;
        ImageViewer view = new ImageViewer("");
        view.show();
        for (DataIterator iter : new DataIterator[] { trainer, validator, tester }) {
            while (iter.hasNextBatch()) {
                for (DataPair data : iter.nextBatch()) {
                    //view.draw(data.input.shapeClone(28, 28), 16);
                    data.input.add(-avg);
                    data.input.product(1. / spread);
                    mean = mean(data.input.getCells());
                    std = sd(data.input.getCells());
                    means.add(mean);
                    stds.add(std);
                    System.out.println("Mean: " + mean + ", std: " + std);
                }
            }
        }
        System.out.println(means.size());
        mean = mean(means.stream().mapToDouble(Double::doubleValue).toArray());
        std = mean(stds.stream().mapToDouble(Double::doubleValue).toArray());
        System.out.println("\r\n\r\nMean: " + mean + ", std: " + std);
    }

    static double mean(double[] ary) {
        double avg = 0;
        int t = 1;
        for (double x : ary) {
            avg += (x - avg) / t;
            ++t;
        }
        return avg;
    }

    public static double sd(double[] ary) {
        // Step 1:
        List<Double> table = DoubleStream.of(ary).boxed().collect(Collectors.toList());

        double mean = mean(ary);
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
}