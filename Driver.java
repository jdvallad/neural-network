
public class Driver {
    // Set hyperparameters here
    static int batchSize = 64;
    static double learningRate = .02;
    static double decayRate = 0.95; // This shrinks the learning rate by 15% each epoch
    static int epochs = 0;
    static int timeToDisplay = 1000;
    static int imageCount = 50;
    static double scale = 10.;
    static NeuralNetwork mnist;
    static DataIterator trainer, validator, tester;
    static String saveLocation = "./serials/cope.ser";

    public static void main(String[] args) throws Exception {
        // Create the network
        // mnist = new NeuralNetwork(28 * 28, "swish");
        // mnist.add(256, "swish");
        // mnist.add(64,"softmax");
        // mnist.add(10);
        // mnist.compile("logLoss");
        mnist = NeuralNetwork.load("./serials/mnistClassify97dot53accurate.ser");
        mnist.printStructure();
        // Initialize the DataIterators
        trainer = new DataIterator(batchSize, "../mnist/training/dataPairs.ser");
        validator = new DataIterator(batchSize, "../mnist/validation/dataPairs.ser");
        tester = new DataIterator(batchSize, "../mnist/testing/dataPairs.ser");

        // This is where the training occurs
        double averageCost = mnist.validate(validator);
        double accuracy = mnist.getClassifierAccuracy(tester);
        mnist.save(saveLocation);
        System.out.println("Initial Statistics:");
        System.out.println("Average Cost: " + averageCost);
        System.out.println("Accuracy: " + accuracy + " %\r\n");
        System.out.println("Batch Size: " + batchSize);
        System.out.println("Number of Batches: " + trainer.numBatches);
        System.out.println("Number of Epochs: " + epochs);
        System.out.println("Initializing training...\r\n");
        long totalStartTime = System.currentTimeMillis();
        for (int i = 1; i < epochs + 1; i++) {
            long epochStartTime = System.currentTimeMillis();
            mnist.train(trainer, learningRate);
            averageCost = mnist.validate(validator);
            accuracy = mnist.getClassifierAccuracy(tester);
            mnist.save(saveLocation);
            long epochEndTime = System.currentTimeMillis();
            double epochTimeSec = (epochEndTime - epochStartTime) / 1000.0;
            System.out.println("Learning Rate: " + learningRate);
            System.out.println("Average Cost: " + averageCost);
            System.out.println("Accuracy: " + accuracy + " %");
            System.out.println("Epoch " + i + " complete in " + epochTimeSec + " seconds\r\n");
            learningRate *= decayRate;
        }
        long totalEndTime = System.currentTimeMillis();
        double totalTimeMin = (totalEndTime - totalStartTime) / 60000.0;
        System.out.println(String.format("Training Complete in %.2f minutes!\r\n", totalTimeMin));
        // Showcase of images and predictions here
        DataPair[] dataPairs = tester.get(0, imageCount);
        ImageViewer viewer = new ImageViewer("inputImage");
        viewer.show();
        for (int i = 0; i < dataPairs.length; i++) {
            viewer.draw(dataPairs[i].input.shapeClone(28, 28), scale);
            Matrix normalizedInput = dataPairs[i].input.clone();
            normalizedInput.minus(tester.mean).divide(tester.standardDeviation);
            int expectedIndex = dataPairs[i].label.maxIndex();
            int actualIndex = mnist.compute(normalizedInput).maxIndex();
            String isWrong = expectedIndex != actualIndex ? " (Incorrect)" : "";
            System.out.println("Expected: " + expectedIndex
                    + ", Output: " + actualIndex + isWrong + "\r\n");
            Thread.sleep(expectedIndex != actualIndex ? timeToDisplay * 2 : timeToDisplay);

        }
        viewer.hide();
        System.exit(0);
    }
}