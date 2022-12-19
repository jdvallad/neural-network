import java.io.*;
import java.util.*;

/**
 * Most up to date version of NeuralNetwork
 */
public class NeuralNetwork implements Serializable {
    double[][][] weights, weightAverages;
    double[][] values, biases, biasAverages;
    String cost, saveFile;
    List<String> activations;
    List<Integer> layers;
    double[][] errors;

    public NeuralNetwork(int inputSize, String cost, String saveFile) {
        activations = new ArrayList<>();
        activations.add("headLayer");
        layers = new ArrayList<>();
        this.cost = cost;
        this.saveFile = saveFile;
        layers.add(inputSize);
    }

    public NeuralNetwork() {
    }

    public NeuralNetwork add(String activation, int out) {
        activations.add(activation);
        layers.add(out);
        return this;
    }

    public NeuralNetwork build() {
        values = new double[layers.size()][];
        biases = new double[layers.size()][];
        weights = new double[layers.size()][][];
        weightAverages = new double[weights.length][][];
        biasAverages = new double[biases.length][];
        values[0] = new double[layers.get(0)];
        biases[0] = new double[layers.get(0)];
        weights[0] = new double[layers.get(0)][];
        for (int r = 1; r < weightAverages.length; r++) {
            values[r] = new double[layers.get(r)];
            biases[r] = new double[layers.get(r)];
            weights[r] = new double[biases[r].length][];
            biasAverages[r] = new double[biases[r].length];
            weightAverages[r] = new double[weights[r].length][];
            for (int c = 0; c < weightAverages[r].length; c++) {
                biases[r][c] = Functions.heParameterInitialize(biases[r - 1].length);
                weights[r][c] = new double[values[r - 1].length];
                weightAverages[r][c] = new double[weights[r][c].length];
                for (int k = 0; k < weightAverages[r][c].length; k++)
                    weights[r][c][k] = Functions.heParameterInitialize(weights[r - 1].length);
            }
        }
        save();
        return this;
    }

    public static NeuralNetwork load(String filePath) {
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
        NeuralNetwork res = new NeuralNetwork();
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
            FileOutputStream fileOut = new FileOutputStream(saveFile);
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(data);
            out.close();
            fileOut.close();
        } catch (IOException i) {
            i.printStackTrace();
        }
    }

    public void showImages(DataIterator iter, Map<String, Object> inputSettings,
            Map<String, Object> outputSettings, int numImages, int timeToDisplay) throws Exception {
        int inputWidth, inputHeight, outputWidth, outputHeight;
        double inputScale, outputScale;
        ImageViewer inputViewer, outputViewer, expectedViewer;
        inputWidth = (int) inputSettings.get("width");
        inputHeight = (int) inputSettings.get("height");
        inputScale = (double) inputSettings.get("scale");
        outputWidth = (int) outputSettings.get("width");
        outputHeight = (int) outputSettings.get("height");
        outputScale = (double) outputSettings.get("scale");
        inputViewer = new ImageViewer("inputViewer");
        outputViewer = new ImageViewer("outputViewer");
        expectedViewer = new ImageViewer("expectedViewer");
        inputViewer.show();
        outputViewer.show();
        expectedViewer.show();
        int counter = 0;
        while (iter.hasNextBatch()) {
            for (DataPair pair : iter.nextBatch()) {
                double[] output = compute(pair.input);
                inputViewer.draw(
                        ImageViewer.listToImage(Functions
                                .scale(Functions.shape(pair.input, inputWidth, inputHeight,
                                        pair.input.length / (inputHeight * inputWidth)), 255.)),
                        inputScale);
                outputViewer.draw(
                        ImageViewer.listToImage(Functions
                                .scale(Functions.shape(output, outputWidth, outputHeight,
                                        output.length / (outputWidth * outputHeight)), 255.)),
                        outputScale);
                expectedViewer.draw(
                        ImageViewer.listToImage(Functions
                                .scale(Functions.shape(pair.expected, outputWidth, outputHeight,
                                        output.length / (outputWidth * outputHeight)),
                                        255.)),
                        outputScale);
                Thread.sleep(timeToDisplay);
                counter++;
                if (counter == numImages) {
                    iter.reset();
                    inputViewer.hide();
                    outputViewer.hide();
                    expectedViewer.hide();
                    return;
                }
            }
        }
        iter.reset();
        inputViewer.hide();
        outputViewer.hide();
        expectedViewer.hide();
        return;
    }

    public void validate(DataIterator validator) throws Exception {
        double totalErrorSum = 0;
        while (validator.hasNextBatch())
            for (DataPair pair : validator.nextBatch())
                totalErrorSum += error(compute(pair.input), pair.expected);
        System.out.println("Average Cost: " + (totalErrorSum / (validator.numBatches * validator.batchSize)));
        validator.reset();
        return;
    }

    public void train(DataIterator trainer, double learningRate) throws Exception {
        resetGradient();
        while (trainer.hasNextBatch()) {
            for (DataPair pair : trainer.nextBatch())
                gradientIncrement(compute(pair.input), pair.expected);
            updateParameters(trainer.batchSize, learningRate);
        }
        trainer.reset();
        return;
    }

    public void getClassifierAccuracy(DataIterator iter) throws Exception {
        double count, correct;
        count = correct = 0;
        while (iter.hasNextBatch()) {
            for (DataPair pair : iter.nextBatch()) {
                double[] output = compute(pair.input);
                count++;
                if (Functions.collapse(output) == Functions.collapse(pair.expected))
                    correct++;
            }
        }
        System.out.println("Total data points: " + (int) count);
        System.out.println("Total missed: " + (int) (count - correct));
        System.out.println("Accuracy: " + (100. * correct / count) + " %");
        iter.reset();
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
        for (int c = 0; c < errors[errors.length - 1].length; c++) {
            if (activations.get(activations.size() - 1).equals("softmax") && cost.equals("logLoss")) {
                errors[errors.length - 1][c] = output[c] - expected[c];
            } else {
                errors[errors.length - 1][c] = Functions.cost(output[c], expected[c], cost, 1)
                        * Functions.activate(weightedSum(errors.length - 1, c),
                                activations.get(activations.size() - 1), 1);
            }
        }
        for (int r = errors.length - 2; r >= 1; r--) {
            errors[r] = new double[biases[r].length];
            for (int c = 0; c < errors[r].length; c++) {
                double sum = 0.;
                for (int i = 0; i < biases[r + 1].length; i++)
                    sum += errors[r + 1][i] * weights[r + 1][i][c];
                sum *= Functions.activate(weightedSum(r, c), activations.get(r), 1);
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
                values[r][c] = Functions.activate(weightedSum(r, c), activations.get(r), 0);
        for (int c = 0; c < biases[biases.length - 1].length; c++)
            if (activations.get(activations.size() - 1).equals("softmax")) {
                values[biases.length - 1][c] = res[c] = Functions.softmax(values[biases.length - 2], weightedSum(biases.length -1, c));
            } else {
                values[biases.length - 1][c] = res[c] = Functions.activate(weightedSum(biases.length - 1, c),
                        activations.get(activations.size() - 1), 0);
            }
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
            sum += Functions.cost(output[i], expected[i], cost, 0);
        return sum / output.length;
    }

    public void printStructure() {
        System.out.print("\r\n---------------------------------");
        for (int r = 0; r < activations.size(); r++) {
            System.out.print("-----");
        }
        System.out.print("\r\nInput layer (" + layers.get(0) + " nodes)");
        int i;
        for (i = 1; i < activations.size() - 1; i++) {
            System.out.print("\r\n");
            for (int r = 0; r < i; r++) {
                System.out.print("----");
            }
            System.out.print("> Hidden layer using " + activations.get(i) + " (" + layers.get(i) + " nodes)");
        }
        System.out.print("\r\n");
        for (int r = 0; r < i; r++) {
            System.out.print("----");
        }
        System.out.print("> Output layer using " + activations.get(activations.size() - 1) + " ("
                + layers.get(layers.size() - 1) + " nodes)\r\n");
        System.out.print("Cost Function: " + cost);
        System.out.print("\r\n---------------------------------");
        for (int r = 0; r < activations.size(); r++) {
            System.out.print("-----");
        }
        System.out.println();
    }

}
