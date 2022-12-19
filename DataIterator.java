import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;

public class DataIterator implements Serializable {
    public final int batchSize;
    public final int numBatches;
    private int batchCounter, index;
    private boolean hasNextBatch;
    private final double[][] data, labels;

    public DataIterator(int batchSize, String filePath) throws IOException, ClassNotFoundException {
        this.batchSize = batchSize;
        batchCounter = 0;
        hasNextBatch = true;
        index = 0;
        double[][][] temp;
        FileInputStream fileIn = new FileInputStream(filePath);
        ObjectInputStream in = new ObjectInputStream(fileIn);
        temp = (double[][][]) in.readObject();
        in.close();
        fileIn.close();
        data = temp[0].clone();
        labels = temp[1].clone();
        this.numBatches = data.length / batchSize;
    }

    public DataPair[] nextBatch() throws Exception {
        DataPair[] res = new DataPair[batchSize];
        for (int i = 0; i < batchSize; i++) {
            res[i] = new DataPair(data[index], labels[index]);
            index++;
        }
        batchCounter++;
        if (batchCounter == numBatches)
            hasNextBatch = false;
        return res;

    }

    public void reset() throws Exception {
        batchCounter = 0;
        hasNextBatch = true;
        index = 0;
    }

    public boolean hasNextBatch() {
        return hasNextBatch;
    }

    public int getBatchCounter() {
        return batchCounter;
    }
}
