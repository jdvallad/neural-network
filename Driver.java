import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class Driver {
    public static void main(String[] args) throws Exception {
        DataIterator nums = new DataIterator(100, 420, true) {
            List<Integer> indices = initializeIndices();

            public List<Integer> initializeIndices() throws Exception {
                List<Integer> temp =
                        Files.lines(Paths.get(
                                "./Data/mnist/train/labels.txt"))
                                .map(Integer::parseInt)
                                .collect(Collectors.toList());
                Collections.shuffle(temp);
                return temp;
            }

            public List<DataPair> nextBatch() throws Exception {
                List<DataPair> res = new ArrayList<>();
                for (int i = 0; i < batchSize; i++) {
                    int index = indices.remove(0);
                    res.add(new DataPair(ImageViewer.imageToList(ImageViewer.pathToImage(formatFilePath(index)), true),
                            formatOutput(index % 10)));
                }
                incrementBatchCounter();
                return res;
            }

            public String formatFilePath(int ind) {
                return "./Data/mnist/train/" + ind + ".jpg";
            }

            public double[] formatOutput(int ind) {
                double[] res = new double[10];
                for (int i = 0; i < 10; i++) {
                    res[i] = (i == ind ? 1. : 0.);
                }
                return res;
            }

            public void reset() throws Exception {
                resetBatchCounter();
                indices = initializeIndices();
            }
        };
        nn model = new nn(28 * 28, "logLoss", "./Data/mnistDigitCategorize.ser");
        model.add("sigmoid", 16);
        model.add("relu", 16);
        model.add("sigmoid", 10);
        //model.build();
        model = nn.load("./Data/mnistDigitCategorize.ser");
        Map<String, Object> input = Map.of("width", 28, "height", 28, "color", false, "scale", 10.);
        Map<String, Object> output = Map.of("width", 10, "height", 1, "color", false, "scale", 80.);
        model.train(nums, 1, .1, true, input, output);
    }
}