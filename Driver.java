import java.util.Map;

//import java.util.Map;

public class Driver {
    public static void main(String[] args) throws Exception {
        final double mnistMean = 46.409632999270976;
        final double mnistSpread = 81.39233883364163;
        DataIterator trainer = new DataIterator(1, "../mnist/training/input.ser", "../mnist/training/output.ser");
        DataIterator validator = new DataIterator(100, "../mnist/validation/input.ser",
                "../mnist/validation/output.ser");
        DataIterator tester = new DataIterator(100, "../mnist/testing/input.ser", "../mnist/testing/output.ser");
        NeuralNetwork mnist = new NeuralNetwork("logLoss", "./serials/mnistDigitClassify.ser")
                .add(28 * 28)
                .add(.8, "leakyRelu", 64)
                .add(.8, "softmax", 10);
        trainer.normalize(mnistMean, mnistSpread);
        validator.normalize(mnistMean, mnistSpread);
        tester.normalize(mnistMean, mnistSpread);
        Map<String, Object> input = Map.of("width", 28, "height", 28, "scale", 10.);
        Map<String, Object> output = Map.of("width", 10, "height", 1, "scale", 80.);
        int epochs = 200;
        mnist.printStructure();
        double learningRate = .001;
        for (int i = 0; i < epochs; i++) {

            mnist.train(trainer, learningRate);
            System.out.println("\r\nEpoch " + (i + 1) + " complete");
            mnist.validate(validator);
            mnist.printClassifierAccuracy(tester);
        }
        // mnist.showImages(tester, input, output, 50, 1000);
        return;
    }
}