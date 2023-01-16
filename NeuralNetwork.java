import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

class NeuralNetwork {

    String cost, saveFile;
    NodeLayer headNode, tailNode;

    public NeuralNetwork(String cost, String saveFile) {
        this.cost = cost;
        this.saveFile = saveFile;
        headNode = null;
        tailNode = null;
    }

    public static NeuralNetwork load(String filePath) {
        // Need to implement
        return null;
    }

    public NeuralNetwork() {
    }

    public NeuralNetwork add(int inputSize) throws Exception {
        if (headNode != null | tailNode != null) {
            throw new Exception("You can only add an input layer once!");
        }
        headNode = new NodeLayer(inputSize);
        tailNode = headNode;
        return this;
    }

    public NeuralNetwork add(String activation, int out) {
        NodeLayer newTail = new NodeLayer(out);
        WeightLayer newWeight = new WeightLayer(tailNode, newTail, activation);
        tailNode.nextWeightLayer = newWeight;
        newTail.previousWeightLayer = newWeight;
        tailNode = newTail;
        return this;
    }

    public void save() {
        // Need to implement
        return;
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
            for (Matrix[] pair : iter.nextBatch()) {
                double[] input = pair[0].getCells();
                double[] expected = pair[1].getCells();
                double[] output = compute(input);
                inputViewer.draw(
                        ImageViewer.listToImage(Functions
                                .scale(Functions.shape(input, inputWidth, inputHeight,
                                        input.length / (inputHeight * inputWidth)), 255.)),
                        inputScale);
                outputViewer.draw(
                        ImageViewer.listToImage(Functions
                                .scale(Functions.shape(output, outputWidth, outputHeight,
                                        output.length / (outputWidth * outputHeight)), 255.)),
                        outputScale);
                expectedViewer.draw(
                        ImageViewer.listToImage(Functions
                                .scale(Functions.shape(expected, outputWidth, outputHeight,
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

    public double[] compute(double[] input) throws Exception {
        NodeLayer temp = headNode;
        Functions.set(temp.values, input);
        while (temp != tailNode) {
            temp.nextWeightLayer.compute();
            temp = temp.nextWeightLayer.nextNodeLayer;
        }
        return temp.values.clone();
    }

    public void validate(DataIterator validator) throws Exception {
        double totalErrorSum = 0;
        while (validator.hasNextBatch())
            for (Matrix[] pair : validator.nextBatch()) {
                double[] input = pair[0].getCells();
                double[] expected = pair[1].getCells();
                totalErrorSum += error(compute(input), expected);
            }
        System.out.println("Average Cost: " + (totalErrorSum / (validator.numBatches * validator.batchSize)));
        validator.reset();
        return;
    }

    public double error(double[] output, double[] expected) throws Exception {
        double sum = 0.;
        for (int i = 0; i < output.length; i++)
            sum += Functions.cost(output[i], expected[i], cost, 0);
        return sum / output.length;
    }

    public void train(DataIterator trainer, double learningRate) throws Exception {
        resetGradient();
        while (trainer.hasNextBatch()) {
            for (Matrix[] pair : trainer.nextBatch()) {
                double[] input = pair[0].getCells();
                double[] expected = pair[1].getCells();
                gradientIncrement(compute(input), expected);
            }
            updateParameters(trainer.batchSize, learningRate);
        }
        trainer.reset();
        return;
    }

    public void resetGradient() {
        NodeLayer temp = headNode;
        while (temp != tailNode) {
            Functions.set(temp.nextWeightLayer.biasAverages, 0);
            Functions.set(temp.nextWeightLayer.weightAverages, 0);
            temp = temp.nextWeightLayer.nextNodeLayer;
        }
    }

    public void gradientIncrement(double[] output, double[] expected) throws Exception {
        NodeLayer temp = tailNode;
        for (int i = 0; i < temp.previousWeightLayer.errors.length; i++) {
            if (temp.previousWeightLayer.activation.equals("softmax") && cost.equals("logLoss")) {
                temp.previousWeightLayer.errors[i] = output[i] - expected[i];
            } else {
                temp.previousWeightLayer.errors[i] = Functions.cost(output[i], expected[i], cost, 1)
                        * Functions.activate(temp.previousWeightLayer.weightedSum(i),
                                temp.previousWeightLayer.activation, 1);
            }
        }
        temp = temp.previousWeightLayer.previousNodeLayer;
        while (temp != headNode) {
            for (int i = 0; i < temp.previousWeightLayer.errors.length; i++) {
                double sum = 0.;
                for (int r = 0; r < temp.nextWeightLayer.biases.length; r++) {
                    sum += temp.nextWeightLayer.errors[r]
                            * temp.nextWeightLayer.weights[i][r];
                }
                sum *= Functions.activate(temp.previousWeightLayer.weightedSum(i), temp.previousWeightLayer.activation,
                        1);
                temp.previousWeightLayer.errors[i] = sum;
            }
            temp = temp.previousWeightLayer.previousNodeLayer;
        }
        temp = tailNode;
        while (temp != headNode) {
            Functions.increase(temp.previousWeightLayer.biasAverages, temp.previousWeightLayer.errors);
            Functions.increase(temp.previousWeightLayer.weightAverages,
                    Functions.multiplyMatrices(
                            Functions.transposeMatrix(
                                    new double[][] { temp.previousWeightLayer.previousNodeLayer.values }),
                            new double[][] { temp.previousWeightLayer.errors }));
            temp = temp.previousWeightLayer.previousNodeLayer;
        }
    }

    public void updateParameters(int batchSize, double learningRate) {
        NodeLayer temp = headNode;
        while (temp != tailNode) {
            if (!temp.nextWeightLayer.locked) {
                Functions.increase(temp.nextWeightLayer.biases,
                        Functions.product(temp.nextWeightLayer.biasAverages, -learningRate / batchSize));
                Functions.increase(temp.nextWeightLayer.weights,
                        Functions.product(temp.nextWeightLayer.weightAverages, -learningRate / batchSize));
            }
            temp = temp.nextWeightLayer.nextNodeLayer;
        }
        resetGradient();
        save();
    }

    public void printStructure() {
        System.out.println("--------------------------------------");
        System.out.println("-->Input layer (" + headNode.numNodes + " nodes)");
        NodeLayer temp = headNode.nextWeightLayer.nextNodeLayer;
        while (temp != tailNode) {
            System.out.println(
                    "-------> Hidden layer using " + temp.previousWeightLayer.activation + " (" + temp.numNodes
                            + " nodes)");
            temp = temp.nextWeightLayer.nextNodeLayer;
        }
        System.out.println("--> Output layer using " + temp.previousWeightLayer.activation + " ("
                + temp.numNodes + " nodes)");
        System.out.println("Cost Function: " + cost);
        System.out.println("--------------------------------------");
    }

    public void getClassifierAccuracy(DataIterator iter) throws Exception {
        double count, correct;
        count = correct = 0;
        while (iter.hasNextBatch()) {
            for (Matrix[] pair : iter.nextBatch()) {
                double[] input = pair[0].getCells();
                double[] expected = pair[1].getCells();
                double[] output = compute(input);
                count++;
                if (Functions.collapse(output) == Functions.collapse(expected))
                    correct++;
            }
        }
        System.out.println("Total data points: " + (int) count);
        System.out.println("Total missed: " + (int) (count - correct));
        System.out.println("Accuracy: " + (100. * correct / count) + " %");
        iter.reset();
    }
}