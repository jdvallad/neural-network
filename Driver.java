import java.util.stream.Stream;

public class Driver {
    // Set hyperparameters here
    static int batchSize = 1;
    static double learningRate = .00001;
    static int epochs = 200;
    static int timeToDisplay = 1000; // 1000 ms, or 1 second
    static int imageCount = 50; // number of images to display
    static double scale = 10.; // used for scaling up image

    public static void main(String[] args) throws Exception {
        // Initialize the network, specifying a save location
        NeuralNetwork mnist = new NeuralNetwork("logLoss", "./serials/mnistDigitClassify.ser");
        mnist.add(28 * 28);
        mnist.add("leakyRelu", 256);
        mnist.add("softmax", 10);
        mnist.printStructure(); // prints an overview of the network to the console

        // Create DataIterators for training, validating, and testing
        DataIterator trainer = new DataIterator(batchSize, "../mnist/training/input.ser",
                "../mnist/training/output.ser");
        DataIterator validator = new DataIterator(batchSize, "../mnist/validation/input.ser",
                "../mnist/validation/output.ser");
        DataIterator tester = new DataIterator(batchSize, "../mnist/testing/input.ser", "../mnist/testing/output.ser");

        // normalize all the data in the DataIterators
        trainer.normalize();
        validator.normalize();
        tester.normalize();

        // this is where the training occurs
        for (int i = 0; i < epochs; i++) {
            mnist.train(trainer, learningRate);
            double averageCost = mnist.validate(validator);
            double accuracy = mnist.getClassifierAccuracy(tester);
            System.out.println("Average Cost: " + averageCost);
            System.out.println("Accuracy: " + accuracy + " %");
            System.out.println("Epoch " + i + " complete\r\n");
        }

        // Showcase of images and predictions here
        ImageViewer viewer = new ImageViewer("inputImage");
        viewer.show();
        trainer.unnormalize(); // puts images back into visible interval
        DataPair[] dataPairs = trainer.getList(imageCount); // get data from trainer iterator
        trainer.normalize();
        Matrix[] input = Stream.of(dataPairs).map(data -> data.input).toArray(Matrix[]::new);
        int[] output = Stream.of(mnist.compute(input)).map(data -> data.maxIndex()).mapToInt(Integer::intValue)
                .toArray();
        int[] labels = Stream.of(dataPairs).map(data -> data.expected.maxIndex()).mapToInt(Integer::intValue)
                .toArray();
        for (int i = 0; i < input.length; i++) {
            viewer.draw(input[i].shape(28, 28), scale);
            System.out.println("Output: " + output[i] + ", Expected: " + labels[i]);
            Thread.sleep(timeToDisplay);
        }
        viewer.hide();
    }
}