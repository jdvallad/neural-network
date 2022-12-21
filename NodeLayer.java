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

}