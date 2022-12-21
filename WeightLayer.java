
public class WeightLayer {
    NodeLayer previousNodeLayer, nextNodeLayer;
    final int inputNodes, outputNodes;
    boolean locked;
    double[] biases, biasAverages, errors;
    double[][] weights, weightAverages;
    String activation;

    public WeightLayer(NodeLayer previousNodeLayer, NodeLayer nextNodeLayer, String activation) {
        this.previousNodeLayer = previousNodeLayer;
        this.nextNodeLayer = nextNodeLayer;
        this.inputNodes = previousNodeLayer.numNodes;
        this.outputNodes = nextNodeLayer.numNodes;
        this.locked = false;
        this.activation = activation;
        this.biases = new double[this.outputNodes];
        this.biasAverages = new double[this.outputNodes];
        this.errors = new double[this.outputNodes];
        this.weights = new double[this.outputNodes][this.inputNodes];
        this.weightAverages = new double[this.outputNodes][this.inputNodes];
        for (int r = 0; r < this.outputNodes; r++) {
            this.biases[r] = Functions.heParameterInitialize(this.previousNodeLayer.numNodes);
            this.weights[r] = new double[inputNodes];
            this.biasAverages[r] = 0;
            this.errors[r] = 0;
            this.weightAverages[r] = new double[inputNodes];
            for (int c = 0; c < this.inputNodes; c++) {
                this.weights[r][c] = Functions.heParameterInitialize(this.previousNodeLayer.numNodes);
                this.weightAverages[r][c] = 0;
            }
        }
        return;
    }

    public void compute() throws Exception {
        for (int r = 0; r < outputNodes; r++) {
            if (this.activation.equals("softmax")) {
                nextNodeLayer.values[r] = Functions.softmax(this.previousNodeLayer.values, weightedSum(r));
            } else {
                nextNodeLayer.values[r] = Functions.activate(weightedSum(r), this.activation, 0);
            }
        }
        return;
    }

    public double weightedSum(int index) {
        double sum = this.biases[index];
        for (int i = 0; i < inputNodes; i++) {
            sum += previousNodeLayer.values[i] * this.weights[index][i];
        }
        return sum;
    }

    public double[] weightedSum() {
        double[] result = new double[outputNodes];
        for (int i = 0; i < result.length; i++) {
            result[i] = weightedSum(i);
        }
        return result;
    }
}