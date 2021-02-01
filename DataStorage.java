import java.io.Serializable;

public class DataStorage implements Serializable {
    public double learningRate;
    double[][][] weights, weightAverage;
    double[][] values, biases, biasAverage;
    String activate, activateLast, cost, saveFile;
}
