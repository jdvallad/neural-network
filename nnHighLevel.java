import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

class nnHighLevel {

    String cost, saveFile;
    NodeLayer headNode, tailNode;

    public nnHighLevel(int inputSize, String cost, String saveFile) {
        this.cost = cost;
        this.saveFile = saveFile;
        headNode = new NodeLayer(inputSize);
        tailNode = headNode;
    }

    public static nnHighLevel load(String filePath) {
        // Need to implement
        return null;
    }

    public nnHighLevel() {
    }

    public nnHighLevel build() {
        return this;
    }

    public nnHighLevel add(String activation, int out) {
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

    public double[] compute(double[] input) throws Exception {
        headNode.feed(input);
        double[] res = headNode.compute();
        return res;
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

    public double error(double[] output, double[] expected) throws Exception {
        double sum = 0.;
        for (int i = 0; i < output.length; i++)
            sum += Functions.cost(output[i], expected[i], cost, 0);
        return sum / output.length;
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

    public void resetGradient() {
        WeightLayer temp = headNode.nextWeightLayer;
        while (temp != null) {
            Functions.set(temp.biasAverages, 0);
            Functions.set(temp.weightAverages, 0);
            temp = temp.nextNodeLayer.nextWeightLayer;
        }
    }

    public void gradientIncrement(double[] output, double[] expected) throws Exception {
        WeightLayer current = tailNode.previousWeightLayer;
        for (int i = 0; i < current.errors.length; i++)
            current.errors[i] = Functions.cost(output[i], expected[i], cost, 1)
                    * Functions.activate(current.weightedSum(i),
                            current.activation, 1);
        current = current.previousNodeLayer.previousWeightLayer;
        while (current != null) {
            for (int i = 0; i < current.errors.length; i++) {
                double sum = 0.;
                for (int r = 0; r < current.nextNodeLayer.nextWeightLayer.biases.length; r++)
                    sum += current.nextNodeLayer.nextWeightLayer.errors[r]
                            * current.nextNodeLayer.nextWeightLayer.weights[r][i];
                sum *= Functions.activate(current.weightedSum(i), current.activation, 1);
                current.errors[i] = sum;
            }
            current = current.previousNodeLayer.previousWeightLayer;
        }
        current = tailNode.previousWeightLayer;
        while (current != null) {
            Functions.increase(current.weightAverages,
                    Functions.multiplyMatrices(Functions.transposeMatrix(new double[][] { current.errors }),
                            new double[][] { current.previousNodeLayer.values }));
            current = current.previousNodeLayer.previousWeightLayer;
        }
    }


    public void updateParameters(int batchSize, double learningRate) {
        WeightLayer temp = headNode.nextWeightLayer;
        while (temp != null) {
            if (!temp.locked) {
                Functions.increase(temp.biases,
                        Functions.product(temp.biasAverages, -learningRate / ((double) batchSize)));
                Functions.increase(temp.weights,
                        Functions.product(temp.weightAverages, -learningRate / ((double) batchSize)));
            }
            temp = temp.nextNodeLayer.nextWeightLayer;
        }
        resetGradient();
        save();
    }

    public void printStructure() {
        System.out.print("\r\n---------------------------------");
        /*
         * for (int r = 0; r < activations.size(); r++) {
         * System.out.print("-----");
         * }
         */
        System.out.print("\r\nInput layer (" + headNode.numNodes + " nodes)");
        NodeLayer temp = headNode.nextWeightLayer.nextNodeLayer;
        while (temp != tailNode) {
            System.out.print("\r\n");
            System.out.print("----");

            System.out.print(
                    "> Hidden layer using " + temp.previousWeightLayer.activation + " (" + temp.numNodes + " nodes)");
            temp = temp.nextWeightLayer.nextNodeLayer;
        }
        System.out.print("\r\n");
        System.out.print("----");
        System.out.print("> Output layer using " + temp.previousWeightLayer.activation + " ("
                + temp.numNodes + " nodes)\r\n");
        System.out.print("Cost Function: " + cost);
        /*
         * System.out.print("\r\n---------------------------------");
         * for (int r = 0; r < activations.size(); r++) {
         * System.out.print("-----");
         * }
         */
        System.out.println();
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
}