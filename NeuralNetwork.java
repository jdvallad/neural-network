import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.List;

class NeuralNetwork {

    String cost, saveFile;
    NodeLayer headNode, tailNode;

    public NeuralNetwork(int inputNodes, String activation) throws Exception {
        headNode = new NodeLayer(inputNodes, activation);
        tailNode = headNode;
    }

    public NeuralNetwork() {

    }

    public void save() throws Exception {
        Map<String, Object> data = new HashMap<>();
        data.put("cost", cost);
        data.put("saveFile", saveFile);
        List<Map<String, Object>> layers = new ArrayList<>();
        NodeLayer temp = headNode;
        while (temp != tailNode) {
            layers.add(temp.save());
            temp = temp.nextNodeLayer;
        }
        layers.add(temp.save());
        data.put("layers", layers);
        FileOutputStream fileOut = new FileOutputStream(saveFile);
        ObjectOutputStream out = new ObjectOutputStream(fileOut);
        out.writeObject(data);
        out.close();
        fileOut.close();
        return;
    }

    public static NeuralNetwork load(String filePath) throws Exception {
        Map<String, Object> data = null;
        FileInputStream fileIn = new FileInputStream(filePath);
        ObjectInputStream in = new ObjectInputStream(fileIn);
        data = (Map<String, Object>) in.readObject();
        in.close();
        fileIn.close();
        NeuralNetwork output = new NeuralNetwork();
        output.cost = (String) data.get("cost");
        output.saveFile = (String) data.get("saveFile");
        List<Map<String, Object>> layers = (List<Map<String, Object>>) data.get("layers");
        NodeLayer prior = NodeLayer.load(layers.get(0));
        output.headNode = prior;
        output.tailNode = prior;
        NodeLayer next = null;
        for (int i = 1; i < layers.size(); i++) {
            next = NodeLayer.load(layers.get(i));
            prior.nextNodeLayer = next;
            next.previousNodeLayer = prior;
            prior = next;
        }
        output.tailNode = next;
        return output;
    }

    public NeuralNetwork add(int inputNodes, String activation) throws Exception {
        NodeLayer newTail = new NodeLayer(inputNodes, activation);
        tailNode.nextNodeLayer = newTail;
        newTail.previousNodeLayer = tailNode;
        tailNode = newTail;
        return this;
    }

    public NeuralNetwork add(int outputNodes) throws Exception {
        if (tailNode == null | tailNode.activation.equals("")) {
            throw new Exception("You can only add an output layer once!");
        }
        return this.add(outputNodes, "");
    }

    public void compile(String cost, String saveFile) throws Exception {
        this.cost = cost;
        this.saveFile = saveFile;
        NodeLayer temp = headNode;
        while (temp != tailNode) {
            temp.compile();
            temp = temp.nextNodeLayer;
        }
    }

    public void printStructure() {
        NodeLayer temp = headNode;
        System.out.println("--------------------------------------");
        System.out.println("-->Input layer with " + temp.numNodes + " nodes using " + temp.activation);
        temp = temp.nextNodeLayer;
        while (temp != tailNode) {
            System.out.println(
                    "-------> Hidden layer with " + temp.numNodes + " nodes using " + temp.activation);
            temp = temp.nextNodeLayer;
        }
        System.out.println("--> Output layer with " + temp.numNodes + " nodes");
        System.out.println("Cost Function: " + cost);
        System.out.println("Save Location: " + saveFile);
        System.out.println("--------------------------------------");
    }

    public Matrix compute(Matrix input) throws Exception {
        NodeLayer temp = headNode;
        temp.feed(input);
        while (temp != tailNode) {
            temp.propogate();
            temp = temp.nextNodeLayer;
        }
        return temp.values.clone();
    }

    public Matrix[] compute(Matrix[] input) throws Exception {
        Matrix[] output = new Matrix[input.length];
        for (int i = 0; i < output.length; i++) {
            output[i] = compute(input[i]);
        }
        return output;
    }

    public void train(DataIterator trainer, double learningRate) throws Exception {
        while (trainer.hasNextBatch()) {
            resetAverages();
            for (DataPair pair : trainer.nextBatch()) {
                Matrix output = compute(pair.input);
                backPropogateErrors(output, pair.label);
                updateAverages();
            }
            updateParameters(-learningRate / (double) trainer.batchSize);
        }
        trainer.reset();
        return;
    }

    public double validate(DataIterator validator) throws Exception {
        double totalErrorSum = 0;
        while (validator.hasNextBatch())
            for (DataPair pair : validator.nextBatch()) {
                totalErrorSum += getCost(compute(pair.input), pair.label);
            }
        double AverageCost = totalErrorSum / (double) (validator.numBatches * validator.batchSize);
        validator.reset();
        return AverageCost;
    }

    public double getClassifierAccuracy(DataIterator iter) throws Exception {
        int count = 0;
        int correct = 0;
        while (iter.hasNextBatch()) {
            for (DataPair pair : iter.nextBatch()) {
                count++;
                if (compute(pair.input).maxIndex() == pair.label.maxIndex()) {
                    correct++;
                }
            }
        }
        iter.reset();
        return 100. * (double) correct / (double) count;
    }

    private double getCost(Matrix output, Matrix expected) throws Exception {
        return output.cost(expected, cost, 0).getSum() / (double) (output.getColumns() * output.getRows());
    }

    private void resetAverages() {
        NodeLayer temp = headNode;
        while (temp != tailNode) {
            temp.biasAverages.zero();
            temp.weightAverages.zero();
            temp = temp.nextNodeLayer;
        }
    }

    private void backPropogateErrors(Matrix output, Matrix expected) throws Exception {
        NodeLayer temp = tailNode;
        temp.backPropogateErrorsTailNode(output, expected, cost);
        temp = temp.previousNodeLayer;
        while (temp != headNode) {
            temp.backPropogateErrors();
            temp = temp.previousNodeLayer;
        }
    }

    private void updateAverages() throws Exception {
        NodeLayer temp = headNode;
        while (temp != tailNode) {
            temp.updateAverages();
            temp = temp.nextNodeLayer;
        }
    }

    private void updateParameters(double scalar) throws Exception {
        NodeLayer temp = headNode;
        while (temp != tailNode) {
            temp.updateParameters(scalar);
            temp = temp.nextNodeLayer;
        }
    }

}