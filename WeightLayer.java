
public class WeightLayer {
    NodeLayer previousNodeLayer, nextNodeLayer;
    final int inputNodes, outputNodes;
    boolean locked;
    Matrix biases, biasAverages, errors;
    Matrix weights, weightAverages;
    String activation;

    public WeightLayer(NodeLayer previousNodeLayer, NodeLayer nextNodeLayer, String activation)
            throws Exception {
        this.previousNodeLayer = previousNodeLayer;
        this.nextNodeLayer = nextNodeLayer;
        this.inputNodes = previousNodeLayer.numNodes;
        this.outputNodes = nextNodeLayer.numNodes;
        this.locked = false;
        this.activation = activation;
        this.biases = Matrix.create(1, this.outputNodes);
        this.biases.heParameterInitialize(this.previousNodeLayer.numNodes);
        this.weights = Matrix.create(this.inputNodes, this.outputNodes);
        this.weights.heParameterInitialize(this.previousNodeLayer.numNodes);
        this.weightAverages = Matrix.create(this.inputNodes, this.outputNodes).zero();
        this.biasAverages = Matrix.create(1, this.outputNodes).zero();
        this.errors = Matrix.create(1, this.outputNodes).zero();
    }

    public void feedForward() throws Exception {
        nextNodeLayer.values.product(previousNodeLayer.values, this.weights).add(this.biases).activate(this.activation,
                0);
    }

    public Matrix weightedSum() throws Exception {
        return Matrix.productClone(previousNodeLayer.values, this.weights).add(this.biases);
    }
}