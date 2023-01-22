import java.util.HashMap;
import java.util.Map;

public class NodeLayer {

    String activation;
    int numNodes;
    boolean locked;
    NodeLayer previousNodeLayer, nextNodeLayer;
    Matrix values, biases, biasAverages, errors, weights, weightAverages;

    public NodeLayer(int nodes, String activation) throws Exception {
        this.nextNodeLayer = null;
        this.previousNodeLayer = null;
        this.numNodes = nodes;
        this.activation = activation;
        this.locked = false;
        this.values = Matrix.create(1, this.numNodes).zero();
    }

    private NodeLayer() {

    }

    public Map<String, Object> save() {
        Map<String, Object> data = new HashMap<>();
        data.put("activation", activation);
        data.put("numNodes", numNodes);
        data.put("locked", locked);
        data.put("values", values.save());
        data.put("biases", biases == null ? null : biases.save());
        data.put("biasAverages", biasAverages == null ? null : biasAverages.save());
        data.put("errors", errors == null ? null : errors.save());
        data.put("weights", weights == null ? null : weights.save());
        data.put("weightAverages", weightAverages == null ? null : weightAverages.save());
        return data;
    }

    public static NodeLayer load(Map<String, Object> data) throws Exception {
        NodeLayer output = new NodeLayer();
        output.activation = (String) data.get("activation");
        output.numNodes = (int) data.get("numNodes");
        output.locked = (boolean) data.get("locked");
        output.values = Matrix.load((Map<String, Object>) data.get("values"));
        output.biases = Matrix.load((Map<String, Object>) data.get("biases"));
        output.biasAverages = Matrix.load((Map<String, Object>) data.get("biasAverages"));
        output.errors = Matrix.load((Map<String, Object>) data.get("errors"));
        output.weights = Matrix.load((Map<String, Object>) data.get("weights"));
        output.weightAverages = Matrix.load((Map<String, Object>) data.get("weightAverages"));
        return output;
    }

    public void compile() throws Exception {
        this.biases = Matrix.create(1, this.nextNodeLayer.numNodes);
        this.biases.heParameterInitialize(this.numNodes);
        this.weights = Matrix.create(this.numNodes, this.nextNodeLayer.numNodes);
        this.weights.heParameterInitialize(this.numNodes);
        this.weightAverages = Matrix.create(this.numNodes, this.nextNodeLayer.numNodes).zero();
        this.biasAverages = Matrix.create(1, this.nextNodeLayer.numNodes).zero();
        this.errors = Matrix.create(1, this.nextNodeLayer.numNodes).zero();
    }

    public void propogate() throws Exception {
        nextNodeLayer.values.product(this.values, this.weights).add(this.biases).activate(this.activation,
                0);
    }

    public void feed(Matrix input) throws Exception {
        this.values.set("*", "*", input);
    }

    public Matrix weightedSum() throws Exception {
        return Matrix.productClone(this.values, this.weights).add(this.biases);
    }

    public void backPropogateErrors() throws Exception {
        this.previousNodeLayer.errors.transpose()
                .innerProduct(this.weights, this.errors).transpose();
        this.previousNodeLayer.errors.elementProduct(this.previousNodeLayer.weightedSum()
                .activate(this.previousNodeLayer.activation, 1));
    }

    public void backPropogateErrorsTailNode(Matrix output, Matrix expected, String cost) throws Exception {
        if (this.previousNodeLayer.activation.equals("softmax") && cost.equals("logLoss")) {
            expected.product(-1);
            this.previousNodeLayer.errors.add(output, expected);
            expected.product(-1);
        } else {
            this.previousNodeLayer.errors.activate(this.previousNodeLayer.weightedSum(),
                    this.previousNodeLayer.activation, 1);
            this.previousNodeLayer.errors.elementProduct(Matrix.costClone(output, expected, cost, 1));
        }
    }

    public void updateAverages() throws Exception {
        this.biasAverages.add(this.errors);
        this.weightAverages.add(Matrix.outerProductClone(this.values, this.errors));
    }

    public void updateParameters(double scalar) throws Exception {
        if (!this.locked) {
            this.biases.add(this.biasAverages.product(scalar));
            this.weights.add(this.weightAverages.product(scalar));
        }
    }
}