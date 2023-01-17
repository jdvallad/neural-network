import java.awt.image.BufferedImage;

//import java.util.Map;

public class Driver {
    public static void main(String[] args) throws Exception {
        ImageViewer view = new ImageViewer("");
        view.show();
        BufferedImage image = ImageViewer.pathToImage("./at.png");
        Matrix a = ImageViewer.getCombinedMatrix(image);
        Matrix red = ImageViewer.getRedMatrix(image);
        Matrix green = ImageViewer.getGreenMatrix(image);
        Matrix blue = ImageViewer.getBlueMatrix(image);
        red.transpose();
        green.transpose();
        blue.transpose();
        view.draw(ImageViewer.combinedMatrixToImage(a.transpose()), .7);
        Thread.sleep(100000);
        long startTime = System.nanoTime();
        DataIterator trainer = new DataIterator(100, "../mnist/training/data.ser");
        DataIterator validator = new DataIterator(100, "../mnist/validation/data.ser");
        DataIterator tester = new DataIterator(100, "../mnist/testing/data.ser");
        NeuralNetwork mnist = new NeuralNetwork("logLoss", "./serials/mnistDigitClassify.ser")
                .add(28 * 28)
                .add("leakyRelu", 32)
                .add("leakyRelu", 32)
                .add("softmax", 10);

        // Map<String, Object> input = Map.of("width", 28, "height", 28, "scale", 10.);
        // Map<String, Object> output = Map.of("width", 10, "height", 1, "scale", 80.);
        int epochs = 20;
        // mnist.printStructure();
        for (int i = 0; i < epochs; i++) {
            mnist.train(trainer, .1);
            System.out.println("Epoch " + (i + 1) + " complete");
            mnist.validate(validator);
        }
        mnist.getClassifierAccuracy(tester);
        // mnist.showImages(tester, input, output, 50, 1000);
        long endTime = System.nanoTime();
        long duration = (endTime - startTime); // divide by 1000000 to get milliseconds.
        System.out.println("This method took " + duration + " nanoseconds.");
        return;
    }
}