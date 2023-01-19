import java.io.FileInputStream;
import java.io.ObjectInputStream;

public class DataIterator {
    public final int batchSize;
    public final int numBatches;
    public final double mean, standardDeviation;

    private boolean isNormalized;

    private int batchCounter, index;
    private boolean hasNextBatch;
    private Matrix[] data, normalizedData, labels;

    public DataIterator(int batchSize, String dataPath, String labelsPath) throws Exception {
        batchCounter = 0;
        hasNextBatch = true;
        index = 0;
        fillDataAndLabels(dataPath, labelsPath);
        if (batchSize == 0) {
            this.batchSize = data.length;
        } else {
            this.batchSize = batchSize;
        }
        this.numBatches = data.length / this.batchSize;
        isNormalized = false;
        isNormalized = false;
        double[] meanAndStandardDeviation = StatisticalTesting.getMeanAndStandardDeviation(data);
        this.mean = meanAndStandardDeviation[0];
        this.standardDeviation = meanAndStandardDeviation[1];
        fillNormalizedData();
    }

    private void fillNormalizedData() throws Exception {
        this.normalizedData = new Matrix[data.length];
        for (int i = 0; i < normalizedData.length; i++) {
            this.normalizedData[i] = data[i].addClone(-mean).product(1. / standardDeviation);
        }
    }

    private void fillDataAndLabels(String dataPath, String labelsPath) throws Exception {
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
    }

    public void normalize() {
        isNormalized = true;
    }

    public void unnormalize() {
        isNormalized = false;
    }

    public DataPair[] nextBatch() throws Exception {
        DataPair[] res = new DataPair[batchSize];
        for (int i = 0; i < batchSize; i++) {
            if (!isNormalized) {
                res[i] = new DataPair(data[index], labels[index]);
            } else {
                res[i] = new DataPair(normalizedData[index], labels[index]);
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

    public DataPair get(int index) {
        if (!isNormalized) {
            return new DataPair(data[index], labels[index]);
        } else {
            return new DataPair(normalizedData[index], labels[index]);
        }
    }

    public DataPair[] getList(int count) {
        DataPair[] output = new DataPair[count];
        for (int i = 0; i < output.length; i++) {
            output[i] = get(i);
        }
        return output;
    }
}
