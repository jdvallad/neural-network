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

    public void add(String activation, int out) {
        NodeLayer newTail = new NodeLayer(out);
        WeightLayer newWeight = new WeightLayer(tailNode, newTail, activation);
        tailNode.nextWeightLayer = newWeight;
        newTail.previousWeightLayer = newWeight;
        tailNode = newTail;
        return;
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
        return headNode.compute();
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
        NodeLayer temp = headNode;
        while(temp != tailNode){
            temp.nextWeightLayer.setBias(0.);
            temp.nextWeightLayer.setWeights(0.);
            temp = temp.nextWeightLayer.nextNodeLayer;
        }
    }

    public void gradientIncrement(double[] output, double[] expected) throws Exception {
        WeightLayer current = tailNode.previousWeightLayer;
        Functions.scale(current.errors, 0);
        for (int i = 0; i < current.errors.length; i++)
            current.errors[i] = Functions.cost(output[i], expected[i], cost, 1)
                    * Functions.activate(current.weightedSum(i), current.weightedSum(),
                            current.activation, 1);
        current = current.previousNLayer.previousWLayer;
        while (current.previousNLayer.previousWLayer != null) {
            Functions.set(current.errors, 0);
            for (int i = 0; i < current.errors.length; i++) {
                double sum = 0.;
                for (int r = 0; r < current.nextNLayer.nextWLayer.biases.length; r++)
                    sum += current.nextNLayer.nextWLayer.errors[r] * current.nextNLayer.nextWLayer.weights[r][i];
                sum *= Functions.activate(current.weightedSum(i), current.weightedSum(), current.activation, 1);
                current.errors[i] = sum;

            }
            current = current.previousNLayer.previousWLayer;
        }


}