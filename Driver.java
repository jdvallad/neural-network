import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class Driver {
public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();
        class mnistIterator extends DataIterator {
            List<Integer> indices = initializeIndices();

            public mnistIterator(int batchSize, int numBatches) throws Exception {
                super(batchSize, numBatches);
            }
        
        
            public List<Integer> initializeIndices() throws Exception {
                List<Integer> temp =
                        Files.lines(Paths.get(
                                "../mnist/train/labels.txt"))
                                .map(Integer::parseInt)
                                .collect(Collectors.toList());
                Collections.shuffle(temp);
                return temp;
            }
        
            public List<DataPair> nextBatch() throws Exception {
                List<DataPair> res = new ArrayList<>();
                for (int i = 0; i < batchSize; i++) {
                    int index = indices.remove(0);
                    res.add(new DataPair(
        
                            imageList(index), mnistLabel(index)
        
                    ));
                }
                incrementBatchCounter();
                return res;
            }
        
            public String formatFilePath(int ind) {
                return "../mnist/train/" + ind + ".jpg";
            }
        
            public double[] imageList(int index) throws Exception {
                return ImageViewer.imageToList(ImageViewer.pathToImage(formatFilePath(index)), false);
            }
        
            public double[] mnistLabel(int ind) {
                double[] res = new double[10];
                for (int i = 0; i < 10; i++) {
                    res[i] = (i == (ind % 10) ? 1. : 0.);
                }
                return res;
            }
        
            public void reset() throws Exception {
                resetBatchCounter();
                indices = initializeIndices();
            }
        }
        DataIterator mnistIter = new mnistIterator(100, 420);
        nn mnist = new nn(28*28, "logLoss", "./serials/abb.ser");
        mnist.add("sigmoid", 16);
        mnist.add("leakyRelu", 16);
        mnist.add("sigmoid", 10);
        mnist.build();
        Map<String, Object> input = Map.of("width", 28, "height", 28, "color", false, "scale", 10.);
        Map<String, Object> output = Map.of("width", 10, "height", 1, "color", false, "scale", 80.);
        mnist.train(mnistIter, 1, .1, true, input, output);
        //mnist.getAccuracy(mnistIter);
        long endTime = System.nanoTime();
        long duration = (endTime - startTime);  //divide by 1000000 to get milliseconds.
        System.out.println("Total Execution Time: " + duration);
        return;
}
}