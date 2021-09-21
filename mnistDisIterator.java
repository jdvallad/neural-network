import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class mnistDisIterator extends DataIterator {
    List<Integer> indices = initializeIndices();
    nn Generator;

    public mnistDisIterator(int batchSize, int numBatches, nn gen) throws Exception {
        super(batchSize, numBatches);
        Generator = gen;
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
         int index;
         for (int i = 0; i < batchSize; i++) {
             if(i % 2 == 0){
                 index = indices.remove(0);
                     res.add(new DataPair(
                         imageList(index), new double[]{1,0}
                     ));
             } else {
                 res.add(new DataPair(

                     Generate.compute(mnistLabel(index)), new double[]{0,1}

                 ));
             }
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
        }
        return res;
    }

    public void reset() throws Exception {
        resetBatchCounter();
        indices = initializeIndices();
    }
}