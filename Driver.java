import java.util.Map;

public class Driver {
    public static void main(String[] args) throws Exception {
        DataIterator trainer = new DataIterator(100, "../mnist/training/data.ser").reverse();
        DataIterator validator = new DataIterator(100, "../mnist/validation/data.ser").reverse();
        DataIterator tester = new DataIterator(100, "../mnist/testing/data.ser").reverse();
        NeuralNetwork mnist = new NeuralNetwork(10, "logLoss", "./serials/mnistDigitGenerate.ser")
                .add("tanh", 16)
                .add("tanh", 16)
                .add("sigmoid", 28*28)
                .build();
        Map<String, Object> output = Map.of("width", 28, "height", 28, "scale", 10.);
        Map<String, Object> input = Map.of("width", 10, "height", 1, "scale", 80.);
        int epochs = 10;
        mnist.printStructure();
        for (int i = 0; i < epochs; i++) {
            mnist.train(trainer, .1);
            System.out.println("Epoch " + (i + 1) + " complete");
            mnist.validate(validator);
        }
        mnist.getClassifierAccuracy(tester);
        mnist.showImages(tester, input, output, 10, 2000);
        return;
    }
}