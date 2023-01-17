import java.awt.image.BufferedImage;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import javax.swing.text.html.ImageView;

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

    public NeuralNetwork add(String activation, int out) throws Exception {
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
        Matrix input, output, expected;
        while (iter.hasNextBatch()) {
            for (DataPair pair : iter.nextBatch()) {
                input = pair.input.shapeClone(inputHeight, inputWidth).product(255.);
                output = compute(pair.input).shapeClone(outputHeight, outputWidth).product(255.);
                expected = pair.expected.shapeClone(outputHeight, outputWidth).product(255.);
                inputViewer.draw(ImageViewer.matrixToImage(input), inputScale);
                outputViewer.draw(ImageViewer.matrixToImage(output), outputScale);
                expectedViewer.draw(ImageViewer.matrixToImage(expected), outputScale);
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

    public Matrix compute(Matrix input) throws Exception {
        NodeLayer temp = headNode;
        temp.feed(input);
        while (temp != tailNode) {
            temp.feedForward();
            temp = temp.nextWeightLayer.nextNodeLayer;
        }
        return temp.values.clone();
    }

    public void validate(DataIterator validator) throws Exception {
        double totalErrorSum = 0;
        while (validator.hasNextBatch())
            for (DataPair pair : validator.nextBatch()) {
                totalErrorSum += error(compute(pair.input), pair.expected);
            }
        System.out.println("Average Cost: " + (totalErrorSum / (validator.numBatches * validator.batchSize)));
        validator.reset();
        return;
    }

    public double error(Matrix output, Matrix expected) throws Exception {
        return output.cost(expected, cost, 0).getSum() / output.getColumns();
    }

    public void train(DataIterator trainer, double learningRate) throws Exception {
        while (trainer.hasNextBatch()) {
            resetAverages();
            for (DataPair pair : trainer.nextBatch()) {
                Matrix output = compute(pair.input);
                updateErrors(output, pair.expected);
                updateAverages(output, pair.expected);
            }
            updateParameters(trainer.batchSize, learningRate);
        }
        trainer.reset();
        return;
    }

    public void resetAverages() {
        NodeLayer temp = headNode;
        while (temp != tailNode) {
            temp.nextWeightLayer.biasAverages.zero();
            temp.nextWeightLayer.weightAverages.zero();
            temp = temp.nextWeightLayer.nextNodeLayer;
        }
    }

    public void updateErrors(Matrix output, Matrix expected) throws Exception {
        NodeLayer temp = tailNode;
        if (temp.previousWeightLayer.activation.equals("softmax") && cost.equals("logLoss")) {
            expected.product(-1);
            temp.previousWeightLayer.errors.add(output, expected);
            expected.product(-1);
        } else {
            temp.previousWeightLayer.errors.activate(temp.previousWeightLayer.weightedSum(),
                    temp.previousWeightLayer.activation, 1);
            temp.previousWeightLayer.errors.elementProduct(Matrix.costClone(output, expected, cost, 1));
        }
        temp = temp.previousWeightLayer.previousNodeLayer;
        while (temp != headNode) {
            temp.previousWeightLayer.errors.transpose()
                    .innerProduct(temp.nextWeightLayer.weights, temp.nextWeightLayer.errors).transpose();
            temp.previousWeightLayer.errors.elementProduct(temp.previousWeightLayer.weightedSum()
                    .activate(temp.previousWeightLayer.activation, 1));
            temp = temp.previousWeightLayer.previousNodeLayer;
        }
        save();
    }

    public void updateAverages(Matrix output, Matrix expected) throws Exception {
        NodeLayer temp = tailNode;
        while (temp != headNode) {
            temp.previousWeightLayer.biasAverages.add(temp.previousWeightLayer.errors);
            temp.previousWeightLayer.weightAverages
                    .add(Matrix.outerProductClone(temp.previousWeightLayer.previousNodeLayer.values,
                            temp.previousWeightLayer.errors));
            temp = temp.previousWeightLayer.previousNodeLayer;
        }
        save();
    }

    public void updateParameters(int batchSize, double learningRate) throws Exception {
        NodeLayer temp = headNode;
        while (temp != tailNode) {
            if (!temp.nextWeightLayer.locked) {
                double scalar = -learningRate / batchSize;
                temp.nextWeightLayer.biases.add(temp.nextWeightLayer.biasAverages.product(scalar));
                temp.nextWeightLayer.weights.add(temp.nextWeightLayer.weightAverages.product(scalar));
            }
            temp = temp.nextWeightLayer.nextNodeLayer;
        }
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

    public void printClassifierAccuracy(DataIterator iter) throws Exception {
        double count, correct;
        count = correct = 0;
        while (iter.hasNextBatch()) {
            for (DataPair pair : iter.nextBatch()) {
                count++;
                if (compute(pair.input).maxIndex() == pair.expected.maxIndex()) {
                    correct++;
                }
            }
        }
        System.out.println("Total data points: " + (int) count);
        System.out.println("Total missed: " + (int) (count - correct));
        System.out.println("Accuracy: " + (100. * correct / count) + " %");
        iter.reset();
    }
}