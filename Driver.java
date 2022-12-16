import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class Driver {
    public static void main(String[] args) throws Exception {
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
        DataIterator mnisIterator = new mnistIterator(100, 420);
        nn model = new nn(28*28, "logLoss", "./serials/a.ser");
        model.add("sigmoid", 16);
        model.add("leakyRelu", 16);
        model.add("sigmoid", 10);
        model.build();
        Map<String, Object> input = Map.of("width", 28, "height", 28, "color", false, "scale", 10.);
        Map<String, Object> output = Map.of("width", 10, "height", 1, "color", false, "scale", 80.);
        //model.train(mnisIterator, 1, .1, true, input, output);
        double[] temp = ImageViewer.imageToList(ImageViewer.pathToImage("./color.jpg"), true);
        ImageViewer img = new ImageViewer("ayo");
        img.show(ImageViewer.listToImage(temp, 821, 547, true));
    }
}