
public class WeightLayer {
    NodeLayer previousNodeLayer, nextNodeLayer;
    final int inputNodes, outputNodes;
    boolean locked;
    Matrix biases, biasAverages, errors;
    Matrix weights, weightAverages;
    String activation;

    public WeightLayer(NodeLayer previousNodeLayer, NodeLayer nextNodeLayer, String activation) throws Exception {
        this.previousNodeLayer = previousNodeLayer;
        this.nextNodeLayer = nextNodeLayer;
        this.inputNodes = previousNodeLayer.numNodes;
        this.outputNodes = nextNodeLayer.numNodes;
        this.locked = false;
        this.activation = activation;
        this.biases = Matrix.create(this.outputNodes, 1).heParameterInitialize(this.previousNodeLayer.numNodes);
        this.weights = Matrix.create(this.inputNodes, this.outputNodes)
                .heParameterInitialize(this.previousNodeLayer.numNodes);
        this.weightAverages = Matrix.create(this.inputNodes, this.outputNodes).zero();
        this.biasAverages = Matrix.create(this.outputNodes, 1).zero();
        this.errors = Matrix.create(this.outputNodes, 1).zero();
        return;
    }

    public void compute() throws Exception {
        nextNodeLayer.values.product(previousNodeLayer.values,this.weights);
        for (int r = 0; r < outputNodes; r++) {
            if (this.activation.equals("softmax")) {
                nextNodeLayer.values[r] = Functions.softmax(this.weightedSum(), weightedSum(r));
            } else {
                nextNodeLayer.values[r] = Functions.activate(weightedSum(r), this.activation, 0);
            }
        }
        return;
    }
}