
public class Driver {
    // Set hyperparameters here
    static int batchSize = 1;
    static double learningRate = .0003;
    static int epochs = 100;

    static int timeToDisplay = 1000;
    static int imageCount = 50;
    static double scale = 10.;
    static NeuralNetwork mnist;
    static DataIterator trainer, validator, tester;
    static String saveLocation = "./serials/yourNeuralNetwork.ser";

    public static void main(String[] args) throws Exception {
        // Create the network
        mnist = new NeuralNetwork(28 * 28, "leakyRelu");
        mnist.add(1024, "softmax");
        mnist.add(10);
        mnist.compile("logLoss");
        mnist.printStructure();

        // Initialize the DataIterators
        trainer = new DataIterator(batchSize, "../mnist/training/dataPairs.ser");
        validator = new DataIterator(batchSize, "../mnist/validation/dataPairs.ser");
        tester = new DataIterator(batchSize, "../mnist/testing/dataPairs.ser");

        // This is where the training occurs
        System.out.println("Initializing training...\r\n");
        for (int i = 1; i < epochs + 1; i++) {
            mnist.train(trainer, learningRate);
            double averageCost = mnist.validate(validator);
            double accuracy = mnist.getClassifierAccuracy(tester);
            mnist.save(saveLocation);
            System.out.println("Average Cost: " + averageCost);
            System.out.println("Accuracy: " + accuracy + " %");
            System.out.println("Epoch " + i + " complete\r\n");
        }
        System.out.println("Training Complete!\r\n");

        // Showcase of images and predictions here
        DataPair[] dataPairs = tester.get(0, imageCount);
        ImageViewer viewer = new ImageViewer("inputImage");
        viewer.show();
        for (int i = 0; i < dataPairs.length; i++) {
            viewer.draw(dataPairs[i].input.shapeClone(28, 28), scale);
            int expectedIndex = dataPairs[i].label.maxIndex();
            int actualIndex = mnist.compute(dataPairs[i].input).maxIndex();
            String isWrong = expectedIndex != actualIndex ? " (Incorrect)" : "";
            System.out.println("Expected: " + expectedIndex
                    + ", Output: " + actualIndex + isWrong + "\r\n");
            Thread.sleep(timeToDisplay);
        }
        viewer.hide();
        return;
    }
}