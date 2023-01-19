import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

public class StatisticalTesting {

    public static double[] getMeanAndStandardDeviation(Matrix[] input) throws Exception {
        List<Double> meanList = new ArrayList<>();
        List<Double> standardDeviationList = new ArrayList<>();
            for (Matrix m : input) {
                meanList.add(getMean(m.getCells()));
                standardDeviationList.add(getStandardDeviation(m.getCells()));
            }
        
        return new double[] { getMean(convertToDoubleArray(meanList)),
                getMean(convertToDoubleArray(standardDeviationList)) };
    }

    private static double[] convertToDoubleArray(List<Double> input) {
        return input.stream().mapToDouble(Double::doubleValue).toArray();
    }

    private static double getMean(double[] ary) {
        double avg = 0;
        int t = 1;
        for (double x : ary) {
            avg += (x - avg) / t;
            ++t;
        }
        return avg;
    }

    private static double getStandardDeviation(double[] ary) {
        List<Double> table = DoubleStream.of(ary).boxed().collect(Collectors.toList());
        double mean = getMean(ary);
        double temp = 0;
        for (int i = 0; i < table.size(); i++) {
            double val = table.get(i);
            double squrDiffToMean = Math.pow(val - mean, 2);
            temp += squrDiffToMean;
        }
        double meanOfDiffs = (double) temp / (double) (table.size());
        return Math.sqrt(meanOfDiffs);
    }
}