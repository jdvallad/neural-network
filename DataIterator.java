import java.io.FileInputStream;
import java.io.ObjectInputStream;

public class DataIterator {
    public final int batchSize;
    public final int numBatches;
    private int batchCounter, index;
    private boolean hasNextBatch;
    private final Matrix[] data, labels;
    private boolean reversed;

    public DataIterator(int batchSize, String dataPath, String labelsPath) throws Exception {
        this.batchSize = batchSize;
        reversed = false;
        batchCounter = 0;
        hasNextBatch = true;
        index = 0;
        double[][] temp;
        FileInputStream fileIn = new FileInputStream(dataPath);
        ObjectInputStream in = new ObjectInputStream(fileIn);
        temp = (double[][]) in.readObject();
        in.close();
        fileIn.close();
        data = new Matrix[temp.length];
        for (int i = 0; i < data.length; i++) {
            data[i] = Matrix.create(temp[i]);
        }
        fileIn = new FileInputStream(labelsPath);
        in = new ObjectInputStream(fileIn);
        temp = (double[][]) in.readObject();
        in.close();
        fileIn.close();
        labels = new Matrix[temp.length];
        for (int i = 0; i < labels.length; i++) {
            labels[i] = Matrix.create(temp[i]);
        }
        this.numBatches = data.length / batchSize;
    }

    public void normalize(double mean, double std) {
        for (int i = 0; i < data.length; i++) {
            data[i].add(-mean).product(1. / std);
        }
        return;
    }

    public void unnormalize(double mean, double std) {
        for (int i = 0; i < data.length; i++) {
            data[i].product(std).add(mean);
        }
        return;
    }

    public DataPair[] nextBatch() throws Exception {
        DataPair[] res = new DataPair[batchSize];
        for (int i = 0; i < batchSize; i++) {
            if (reversed) {
                res[i] = new DataPair(labels[index], data[index]);
            } else {
                res[i] = new DataPair(data[index], labels[index]);
            }
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

    public DataIterator reverse() {
        reversed = !reversed;
        return this;
    }
}
