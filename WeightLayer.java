
public class WeightLayer {
    NodeLayer previousNodeLayer, nextNodeLayer;
    final int inputNodes, outputNodes;
    boolean locked;
    Matrix biases, biasAverages, errors;
    Matrix weights, weightAverages;
    String activation;
    double keepProbability;

    public WeightLayer(NodeLayer previousNodeLayer, NodeLayer nextNodeLayer, String activation, double keepProbability)
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
        this.keepProbability = keepProbability;
    }

    public void feedForward() throws Exception {
        nextNodeLayer.values.product(previousNodeLayer.values, this.weights).add(this.biases).activate(this.activation,
                0);
    }

    public Matrix weightedSum() throws Exception {
        return Matrix.productClone(previousNodeLayer.values, this.weights).add(this.biases);
    }

    public void dropout() throws Exception {
        for (int r = 0; r < this.previousNodeLayer.values.getRows(); r++) {
            for (int c = 0; c < this.previousNodeLayer.values.getColumns(); c++) {
                if (Math.random() > this.keepProbability) {
                    this.previousNodeLayer.values.set(r, c, 0);
                }
            }
        }
    }

    public void dropoutUpscale() throws Exception {
        this.previousNodeLayer.values.product(keepProbability);
    }

    public void dropoutDownscale() throws Exception {
        this.previousNodeLayer.values.product(1. / keepProbability);
    }
}