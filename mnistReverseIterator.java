import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class mnistReverseIterator extends DataIterator {
    List<Integer> indices = initializeIndices();

    public mnistReverseIterator(int batchSize, int numBatches) throws Exception {
        super(batchSize, numBatches);
    }


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
            res.add(new DataPair(

                    mnistLabel(index), imageList(index)

            ));
        }
        incrementBatchCounter();
        return res;
    }

    public String formatFilePath(int ind) {
        return "./Data/mnist/train/" + ind + ".jpg";
    }

    public double[] imageList(int index) throws Exception {
        return ImageViewer.imageToList(ImageViewer.pathToImage(formatFilePath(index)), true);
    }

    public double[] mnistLabel(int ind) {
        double[] res = new double[10];
        for (int i = 0; i < 10; i++) {
            res[i] = (i == (ind % 10) ? 1. : 0.);
            res[i] += Math.random() * .1;
        }
        return res;
    }

    public void reset() throws Exception {
        resetBatchCounter();
        indices = initializeIndices();
    }
}