import java.io.*;
import java.util.*;

/**
 * Newest version of Neural Network
 * actually god tier
 * no extra objects, only primitives
 * change network parameters in the function class
 * if need be, change how the image outputs are printed, not necessary
 */
public class nn implements Serializable {
    public double learningRate = .1;
    double[][][] weights, weightAverage;
    double[][] values, biases, biasAverage;
    String activate, activateLast, cost, saveFile;

    public nn(String x, String y, String z, String str, int... layers) {
        activate = x;
        activateLast = y;
        cost = z;
        saveFile = str;
        values = new double[layers.length][];
        biases = new double[layers.length][];
        weights = new double[layers.length][][];
        weightAverage = new double[weights.length][][];
        biasAverage = new double[biases.length][];
        values[0] = new double[layers[0]];
        for (int r = 1; r < weightAverage.length; r++) {
            values[r] = new double[layers[r]];
            biases[r] = new double[layers[r]];
            weights[r] = new double[biases[r].length][];
            biasAverage[r] = new double[biases[r].length];
            weightAverage[r] = new double[weights[r].length][];
            int len = values[r - 1].length;
            for (int c = 0; c < weightAverage[r].length; c++) {
                biases[r][c] = Functions.biasInitialize();
                weights[r][c] = new double[len];
                weightAverage[r][c] = new double[weights[r][c].length];
                for (int k = 0; k < weightAverage[r][c].length; k++)
                    weights[r][c][k] = Functions.weightInitialize();
            }
        }
        save();
    }

    public nn() {
    }

    public static nn load(String filePath) {
        DataStorage data = null;
        try {
            FileInputStream fileIn = new FileInputStream(filePath);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            data = (DataStorage) in.readObject();
            in.close();
            fileIn.close();
        } catch (IOException | ClassNotFoundException i) {
            i.printStackTrace();
        }
        nn res = new nn();
        assert data != null;
        res.activate = data.activate;
        res.activateLast = data.activateLast;
        res.cost = data.cost;
        res.saveFile = data.saveFile;
        res.learningRate = data.learningRate;
        res.weights = data.weights;
        res.weightAverage = data.weightAverage;
        res.values = data.values;
        res.biases = data.biases;
        res.biasAverage = data.biasAverage;
        return res;
    }

    public void save() {
        DataStorage data = new DataStorage();
        data.activate = activate;
        data.activateLast = activateLast;
        data.cost = cost;
        data.saveFile = saveFile;
        data.learningRate = learningRate;
        data.weights = weights;
        data.weightAverage = weightAverage;
        data.values = values;
        data.biases = biases;
        data.biasAverage = biasAverage;
        try {
            FileOutputStream fileOut =
                    new FileOutputStream(saveFile);
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(data);
            out.close();
            fileOut.close();
        } catch (IOException i) {
            i.printStackTrace();
        }
    }

    public void train(DataIterator data, int epochs) throws Exception {
        double[] input;
        double[] output;
        double[] expected;
        double errorAverage = 0;
        for (int i = 0; i < epochs; i++) {
            while (data.hasNextBatch()) {
                resetGradient();
                errorAverage = 0.;
                for (IEPair pair : data.nextBatch()) {
                    input = pair.input;
                    expected = pair.expected;
                    output = compute(input);
                    gradientIncrement(output, expected);
                    errorAverage += error(output, expected);
                }
                updateParameters(data.batchSize);
                save();
                //System.out.println("Cost: " + (errorAverage / (double) data.batchSize));
            }
            System.out.println("Epoch " + (i + 1)+"\r\n\tCost: "  + (errorAverage / (double) data.batchSize));
            data.reset();
        }
    }

    public void resetGradient() {
        for (int r = 1; r < weightAverage.length; r++) {
            for (int c = 0; c < weightAverage[r].length; c++) {
                biasAverage[r][c] = 0.;
                Arrays.fill(weightAverage[r][c], 0.);
            }
        }
    }

    public void updateParameters(int batchSize) {
        for (int r = 1; r < weights.length; r++) {
            for (int c = 0; c < weights[r].length; c++) {
                biases[r][c] -= (learningRate * biasAverage[r][c]) / batchSize;
                for (int k = 0; k < weights[r][c].length; k++)
                    weights[r][c][k] -= (learningRate * weightAverage[r][c][k]) / batchSize;
            }
        }
    }

    public void gradientIncrement(double[] output, double[] expected) throws Exception {
        double[][] errors = new double[biases.length][];
        errors[errors.length - 1] = new double[biases[biases.length - 1].length];
        for (int c = 0; c < errors[errors.length - 1].length; c++)
            errors[errors.length - 1][c] = Functions.cost(output[c], expected[c], cost, false) * Functions.activateLastLayer(weightedSum(errors.length - 1, c), activateLast, false);
        for (int r = errors.length - 2; r >= 1; r--) {
            errors[r] = new double[biases[r].length];
            for (int c = 0; c < errors[r].length; c++) {
                double sum = 0.;
                for (int i = 0; i < biases[r + 1].length; i++)
                    sum += errors[r + 1][i] * weights[r + 1][i][c];
                sum *= Functions.activate(weightedSum(r, c), activate, false);
                errors[r][c] = sum;
            }
        }
        for (int r = biases.length - 1; r >= 1; r--) {
            for (int c = 0; c < biases[r].length; c++) {
                biasAverage[r][c] += errors[r][c];
                for (int k = 0; k < weights[r][c].length; k++)
                    weightAverage[r][c][k] += errors[r][c] * values[r - 1][k];
            }
        }
    }

    public double[] compute(double[] input) throws Exception {
        double[] res = new double[biases[biases.length - 1].length];
        System.arraycopy(input, 0, values[0], 0, values[0].length);
        for (int r = 1; r < biases.length - 1; r++)
            for (int c = 0; c < biases[r].length; c++)
                values[r][c] = Functions.activate(weightedSum(r, c), activate, true);
        for (int c = 0; c < biases[biases.length - 1].length; c++)
            values[biases.length - 1][c] = res[c] = Functions.activateLastLayer(weightedSum(biases.length - 1, c), activateLast, true);
        return res;
    }

    public double weightedSum(int a, int b) {
        double sum = biases[a][b];
        for (int c = 0; c < values[a - 1].length; c++)
            sum += values[a - 1][c] * weights[a][b][c];
        return sum;
    }

    public double error(double[] output, double[] expected) throws Exception {
        double sum = 0.;
        for (int i = 0; i < output.length; i++)
            sum += Functions.cost(output[i], expected[i], cost, true);
        return sum / output.length;
    }
}
