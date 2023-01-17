public class NodeLayer {
    WeightLayer previousWeightLayer, nextWeightLayer;
    Matrix values;
    final int numNodes;

    public NodeLayer(int nodes) throws Exception {
        this.nextWeightLayer = null;
        this.previousWeightLayer = null;
        this.numNodes = nodes;
        this.values = Matrix.create(1, nodes).zero();

    }

    public void feedForward() throws Exception {
        this.nextWeightLayer.feedForward();
    }

    public void feed(Matrix input) throws Exception {
        this.values.set("*", "*", input);
    }
}