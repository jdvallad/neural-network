import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Layer implements Serializable {
    public String activation;
    public int inputSize, outputSize;
    public double[] biases;
    public double[][] weights;
    public double[] output;
    public double[] biasGradients;
    public double[][] weightGradients;
    public double[] zeds;
    public Layer priorLayer;
    public Layer nextLayer;

    public Layer(int inputSize, int outputSize, String activation) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activation = activation;
        biases = new double[outputSize];
        weights = new double[outputSize][];
        biasGradients = new double[outputSize];
        weightGradients = new double[outputSize][];
        for (int i = 0; i < outputSize; i++) {
            biases[i] = 100. * Math.random() - 50.;
            weights[i] = new double[inputSize];
            weightGradients[i] = new double[inputSize];
            for (int k = 0; k < inputSize; k++)
                weights[i][k] = 100. * Math.random() - 50.;
        }
        output = new double[outputSize];
        zeds = new double[outputSize];
        priorLayer = nextLayer = null;
    }

    public void resetGradients() {
        for (int i = 0; i < outputSize; i++) {
            biasGradients[i] = 0.;
            for (int k = 0; k < inputSize; k++)
                weightGradients[i][k] = 0.;
        }
    }

    public double[] feedForward(double[] input, boolean train) {
        Helper.clear(output, 0.);
        for (int i = 0; i < outputSize; i++) {
            zeds[i] = Helper.zed(input, weights[i], biases[i]);
            output[i] = Helper.sigmoid(zeds[i], false);
        }
        if (train) {
            double[][] w = w(input, output);
            double[] b = b(input, output);
            for (int j = 0; j < outputSize; j++) {
                biasGradients[j] += b[j];
                for (int k = 0; k < inputSize; k++)
                    weightGradients[j][k] += w[j][k];
            }
        }
        return output.clone();
    }

    public double w(int j, int k, double[] input, double[] expected) {
        double a = input[k];
        double b = Helper.sigmoid(zeds[j], true);
        double c = func(j, expected);
        return a * b * c;
    }

    public double b(int j, double[] input, double[] expected) {
        double b = Helper.sigmoid(zeds[j], true);
        double c = func(j, expected);
        return b * c;
    }

    public double[] b(double[] input, double[] expected) {
        double[] res = new double[outputSize];
        for (int j = 0; j < outputSize; j++)
            res[j] = b(j, input, expected);
        return res;
    }

    public double[][] w(double[] input, double[] expected) {
        double[][] res = new double[outputSize][inputSize];
        for (int j = 0; j < outputSize; j++)
            for (int k = 0; k < inputSize; k++)
                res[j][k] = w(j, k, input, expected);
        return res;
    }

    public double func(int k, double[] expected) {
        if (nextLayer == null)
            return 2. * (output[k] - expected[k]);
        else {
            double sum = 0.;
            for (int j = 0; j < nextLayer.outputSize; j++) {
                double a = weights[j][k];
                double b = Helper.sigmoid(nextLayer.zeds[j], true);
                double c = nextLayer.func(j, expected);
                sum += a * b * c;
            }
            return sum;
        }
    }

    public double[] func(double[] expected) {
        double[] res = new double[outputSize];
        for (int k = 0; k < outputSize; k++)
            res[k] = func(k, expected);
        return res;
    }

    public void nudge(double[] weightShifts, double biasShift, int index, double scale) {
        biases[index] += scale * biasShift;
        for (int i = 0; i < weights[index].length; i++)
            weights[index][i] += scale * weightShifts[i];
    }

    public void nudge(double[][] weightShifts, double[] biasShifts, double scale) {
        for (int i = 0; i < outputSize; i++)
            nudge(weightShifts[i], biasShifts[i], i, scale);
    }
}
