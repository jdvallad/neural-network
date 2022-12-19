public class NodeLayer {
    WeightLayer previousWeightLayer, nextWeightLayer;
    double[] values;
     final int numNodes;

    public NodeLayer(int nodes) {
        this.nextWeightLayer = null;
        this.previousWeightLayer = null;
        this.numNodes = nodes;
        this.values = new double[nodes];
        for (int i = 0; i < this.numNodes; i++) {
            this.values[i] = 0;
        }
    }

    public NodeLayer(NodeLayer n) {
        this.numNodes = n.numNodes;
        this.nextWeightLayer = n.nextWeightLayer;
        this.previousWeightLayer = n.previousWeightLayer;
        this.values = new double[numNodes];
        for (int i = 0; i < this.numNodes; i++) {
            this.values[i] = n.values[i];
        }
    }

    public void feed(double[] input) {
        for (int i = 0; i < this.numNodes; i++) {
            this.values[i] = input[i];
        }
    }

}