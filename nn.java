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
    double[][][] weights, weightAverages;
    double[][] values, biases, biasAverages;
    String cost, saveFile;
    List<String> activations;
    List<Integer> layers;
    double[][] errors;

    public nn(int inputSize, String cost, String saveFile) {
        activations = new ArrayList<>();
        activations.add("TEMP");
        layers = new ArrayList<>();
        this.cost = cost;
        this.saveFile = saveFile;
        layers.add(inputSize);
    }

    public nn() {
    }

    public void add(String activation, int out) {
        activations.add(activation);
        layers.add(out);
    }

    public void build() {
        values = new double[layers.size()][];
        biases = new double[layers.size()][];
<<<<<<< HEAD
=======
        locked = new ArrayList<Boolean>();
        locked.add(null);
>>>>>>> 01b4f0ce2f5e37f1aece051fe35dd4e044fb1b0a
        weights = new double[layers.size()][][];
        weightAverages = new double[weights.length][][];
        biasAverages = new double[biases.length][];
        values[0] = new double[layers.get(0)];
        for (int r = 1; r < weightAverages.length; r++) {
            values[r] = new double[layers.get(r)];
            biases[r] = new double[layers.get(r)];
            weights[r] = new double[biases[r].length][];
            biasAverages[r] = new double[biases[r].length];
            weightAverages[r] = new double[weights[r].length][];
<<<<<<< HEAD
=======
            this.locked.add(false);
>>>>>>> 01b4f0ce2f5e37f1aece051fe35dd4e044fb1b0a
            for (int c = 0; c < weightAverages[r].length; c++) {
                biases[r][c] = Functions.biasInitialize();
                weights[r][c] = new double[values[r - 1].length];
                weightAverages[r][c] = new double[weights[r][c].length];
                for (int k = 0; k < weightAverages[r][c].length; k++)
                    weights[r][c][k] = Functions.weightInitialize();
            }
        }
        save();
    }

    public static nn load(String filePath) {
        Map<String, Object> data = null;
        try {
            FileInputStream fileIn = new FileInputStream(filePath);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            data = (Map<String, Object>) in.readObject();
            in.close();
            fileIn.close();
        } catch (IOException | ClassNotFoundException i) {
            i.printStackTrace();
        }
        nn res = new nn();
        assert data != null;
        res.cost = (String) data.get("cost");
        res.saveFile = (String) data.get("saveFile");
        res.weights = (double[][][]) data.get("weights");
        res.weightAverages = (double[][][]) data.get("weightAverages");
        res.values = (double[][]) data.get("values");
        res.biases = (double[][]) data.get("biases");
        res.biasAverages = (double[][]) data.get("biasAverages");
        res.activations = (List<String>) data.get("activations");
        return res;
    }

    public void save() {
        Map<String, Object> data = new HashMap<>();
        data.put("cost", cost);
        data.put("saveFile", saveFile);
        data.put("weights", weights);
        data.put("weightAverages", weightAverages);
        data.put("values", values);
        data.put("biases", biases);
        data.put("biasAverages", biasAverages);
        data.put("activations", activations);
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

    public void train(DataIterator data, int epochs, double learningRate, boolean print) throws Exception {
        resetGradient();
        double batchErrorSum, totalErrorSum;
        for (int i = 0; i < epochs; i++) {
            totalErrorSum = 0;
            while (data.hasNextBatch()) {
                batchErrorSum = 0;
                for (DataPair pair : data.nextBatch()) {
                    double[] output = compute(pair.input);
                    gradientIncrement(output, pair.expected);
                    if (print)
                        batchErrorSum += error(output, pair.expected);
                }
                if (print)
                    System.out.println("Batch Average Cost: " + batchErrorSum / data.batchSize);
                totalErrorSum += batchErrorSum / data.batchSize;
                updateParameters(data.batchSize, learningRate);
            }
            if (print)
                System.out.println("\r\n\r\nEpoch " + (i + 1) + " Average Cost: " + (totalErrorSum / data.numBatches) + "\r\n\r\n");
            System.out.println("\r\nEpoch " + (i + 1) + " complete.\r\n");
            data.reset();
        }
    }

    public void train(DataIterator data, int epochs, double learningRate, boolean print, Map<String, Object> input, Map<String, Object> expected) throws Exception {
        int inputWidth = (int) input.get("width");
        int inputHeight = (int) input.get("height");
        double inputScale = (double) input.get("scale");
        boolean inputColor = (boolean) input.get("color");
        int expectedWidth = (int) expected.get("width");
        int expectedHeight = (int) expected.get("height");
        double expectedScale = (double) expected.get("scale");
        boolean expectedColor = (boolean) expected.get("color");
        ImageViewer one, two, three;
        one = new ImageViewer("input");
        two = new ImageViewer("output");
        three = new ImageViewer("expected");
        one.setVisible(true);
        two.setVisible(true);
        three.setVisible(true);
        resetGradient();
        double batchErrorSum, totalErrorSum;
        for (int i = 0; i < epochs; i++) {
            totalErrorSum = 0;
            while (data.hasNextBatch()) {
                batchErrorSum = 0;
                for (DataPair pair : data.nextBatch()) {
                    double[] output = compute(pair.input);
                    gradientIncrement(output, pair.expected);
                    if (print)
                        batchErrorSum += error(output, pair.expected);
                    one.show(ImageViewer.listToImage(pair.input, inputWidth, inputHeight, inputColor), inputScale);
                    two.show(ImageViewer.listToImage(output, expectedWidth, expectedHeight, expectedColor), expectedScale);
                    three.show(ImageViewer.listToImage(pair.expected, expectedWidth, expectedHeight, expectedColor), expectedScale);
                }
                if (print)
                    System.out.println("Batch Average Cost: " + batchErrorSum / data.batchSize);
                totalErrorSum += batchErrorSum / data.batchSize;
                updateParameters(data.batchSize, learningRate);
            }
            if (print)
                System.out.println("\r\n\r\nEpoch " + (i + 1) + " Average Cost: " + (totalErrorSum / data.numBatches) + "\r\n\r\n");
            System.out.println("\r\nEpoch " + (i + 1) + " complete.\r\n");
            data.reset();
        }
        one.setVisible(false);
        two.setVisible(false);
        three.setVisible(false);
    }

    public void getAccuracy(DataIterator data) throws Exception {
        double count, correct;
        count = correct = 0;
        while (data.hasNextBatch()) {
            for (DataPair pair : data.nextBatch()) {
                double[] output = compute(pair.input);
                count++;
                if (Functions.collapse(output) == Functions.collapse(pair.expected))
                    correct++;
            }
            System.out.println("Accuracy: " + (100. * correct / count) + " %\r\n");
        }
        System.out.println("Total data points: " + (int) count);
        System.out.println("Total missed: " + (int) (count - correct));
        System.out.println("Final Accuracy: " + (100. * correct / count) + " %");
    }

    public void resetGradient() {
        for (int r = 1; r < weightAverages.length; r++) {
            for (int c = 0; c < weightAverages[r].length; c++) {
                biasAverages[r][c] = 0.;
                Arrays.fill(weightAverages[r][c], 0.);
            }
        }
    }

    public void updateParameters(int batchSize, double learningRate) {
        for (int r = 1; r < weights.length; r++) {
            for (int c = 0; c < weights[r].length; c++) {
                biases[r][c] -= (learningRate * biasAverages[r][c]) / batchSize;
                for (int k = 0; k < weights[r][c].length; k++)
                    weights[r][c][k] -= (learningRate * weightAverages[r][c][k]) / batchSize;
            }
        }
        resetGradient();
        save();
    }

    public void gradientIncrement(double[] output, double[] expected) throws Exception {
        errors = new double[biases.length][];
        errors[errors.length - 1] = new double[biases[biases.length - 1].length];
        for (int c = 0; c < errors[errors.length - 1].length; c++)
            errors[errors.length - 1][c] = Functions.cost(output[c], expected[c], cost, false)
                    * Functions.activate(weightedSum(errors.length - 1, c),
                    activations.get(activations.size() - 1), false);
        for (int r = errors.length - 2; r >= 1; r--) {
            errors[r] = new double[biases[r].length];
            for (int c = 0; c < errors[r].length; c++) {
                double sum = 0.;
                for (int i = 0; i < biases[r + 1].length; i++)
                    sum += errors[r + 1][i] * weights[r + 1][i][c];
                sum *= Functions.activate(weightedSum(r, c), activations.get(r), false);
                errors[r][c] = sum;
            }
        }
        for (int r = biases.length - 1; r >= 1; r--) {
            for (int c = 0; c < biases[r].length; c++) {
                biasAverages[r][c] += errors[r][c];
                for (int k = 0; k < weights[r][c].length; k++)
                    weightAverages[r][c][k] += errors[r][c] * values[r - 1][k];
            }
        }
    }

    public double[] compute(double[] input) throws Exception {
        double[] res = new double[biases[biases.length - 1].length];
        System.arraycopy(input, 0, values[0], 0, values[0].length);
        for (int r = 1; r < biases.length - 1; r++)
            for (int c = 0; c < biases[r].length; c++)
                values[r][c] = Functions.activate(weightedSum(r, c), activations.get(r), true);
        for (int c = 0; c < biases[biases.length - 1].length; c++)
            values[biases.length - 1][c] = res[c]
                    = Functions.activate(weightedSum(biases.length - 1, c)
                    , activations.get(activations.size() - 1), true);
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
