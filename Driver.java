import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class Driver {
    public static void main(String[] args) throws Exception {
        //  nn network = new nn("sigmoid", "sigmoid", "meanSquaredError", "C:\\Users\\Joshua Valladares\\Desktop\\Projects\\Data\\nn.ser", 28*28, 16,16,10);
        nn network = nn.load("C:\\Users\\Joshua Valladares\\Desktop\\Projects\\Data\\nn.ser");
        DataIterator numbers = numbers();
        network.train(numbers, 5);
    }

    public static int answer(double[] arr) {
        int max = 0;
        for (int i = 0; i < arr.length; i++)
            if (arr[i] > arr[max])
                max = i;
        return max;
    }

    public static DataIterator numbers() throws Exception {
        return new DataIterator(100, 420, true) {
            List<Integer> indices = initializeIndices();

            public List<Integer> initializeIndices() throws Exception {
                List<Integer> temp =
                        Files.lines(Paths.get(
                                "C:\\Users\\Joshua Valladares\\Desktop\\Projects" +
                                        "\\Data\\mnist\\trainingSet\\labels.txt"))
                                .map(Integer::parseInt)
                                .collect(Collectors.toList());
                Collections.shuffle(temp);
                return temp;
            }

            public List<IEPair> nextBatch() throws Exception {
                List<IEPair> res = new ArrayList<>();
                for (int i = 0; i < batchSize; i++) {
                    int index = indices.remove(0);
                    res.add(
                            new IEPair(
                                    Helper.convertColorImageToDoubleList(
                                            Helper.imageFromFilePath(
                                                    formatFilePath(index))),
                                    formatOutput(index % 10)));
                }
                incrementBatchCounter();
                return res;
            }

            public String formatFilePath(int ind) {
                return "C:\\Users\\Joshua Valladares\\Desktop\\Projects\\Data\\mnist\\trainingSet\\images\\" + ind + ".jpg";
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
    }

    public static DataIterator faces() {
        return new DataIterator(100, 2000, false) {
            List<Integer> indices = formatIndices();

            public List<IEPair> nextBatch() throws Exception {
                List<IEPair> res = new ArrayList<>();
                for (int i = 0; i < batchSize; i++)
                    res.add(new IEPair(noise(), Helper.convertColorImageToDoubleList(Helper.imageFromFilePath(formatFilePath(indices.remove(0))))));
                incrementBatchCounter();
                return res;
            }

            public double[] noise() {
                double[] res = new double[10];
                for (int i = 0; i < 10; i++)
                    res[i] = Math.random();
                return res;
            }

            public List<Integer> formatIndices() {
                List<Integer> res = new ArrayList<>();
                for (int i = 1; i <= 162770; i++)
                    res.add(i);
                Collections.shuffle(res);
                return res;
            }

            public String formatFilePath(int ind) {
                String res = "000000" + ind;
                res = res.substring(res.length() - 6);
                return "C:\\Users\\Joshua Valladares\\Desktop\\Projects\\Data\\faces\\" + res + ".jpg";
            }

            @Override
            public void reset() throws Exception {
                indices = formatIndices();
                resetBatchCounter();
            }
        };
    }
}