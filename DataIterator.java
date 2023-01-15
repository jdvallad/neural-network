import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

public class DataIterator {
    public final int batchSize;
    public final int numBatches;
    private int batchCounter, index;
    private boolean hasNextBatch;
    private final Matrix[] data, labels;
    private boolean reversed;

    public DataIterator(int batchSize, String filePath) throws Exception {
        this.batchSize = batchSize;
        reversed = false;
        batchCounter = 0;
        hasNextBatch = true;
        index = 0;
        double[][][] temp;
        FileInputStream fileIn = new FileInputStream(filePath);
        ObjectInputStream in = new ObjectInputStream(fileIn);
        temp = (double[][][]) in.readObject();
        in.close();
        fileIn.close();
        data = new Matrix[temp[0].length];
        for (int i = 0; i < data.length; i++) {
            data[i] = Matrix.create(temp[0][i]);
        }
        labels = new Matrix[temp[1].length];
        for (int i = 0; i < labels.length; i++) {
            labels[i] = Matrix.create(temp[1][i]);
        }
        this.numBatches = data.length / batchSize;
    }

    public Matrix[][] nextBatch() throws Exception {
        Matrix[][] res = new Matrix[batchSize][];
        for (int i = 0; i < batchSize; i++) {
            if (reversed) {
                res[i] = new Matrix[] { labels[index], data[index] };
            } else {
                res[i] = new Matrix[] { data[index], labels[index] };
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
