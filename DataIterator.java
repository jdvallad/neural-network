import java.io.IOException;
import java.io.Serializable;
import java.util.List;

public abstract class DataIterator implements Serializable {
    protected final int batchSize;
    protected final int numBatches;
    private int batchCounter;
    private boolean hasNextBatch;
    protected final boolean hasExpected;

    public DataIterator(int batchSize, int numBatches, boolean hasExpected) {
        this.batchSize = batchSize;
        this.numBatches = numBatches;
        this.hasExpected = hasExpected;
        batchCounter = 0;
        hasNextBatch = true;
    }

    public abstract List<IEPair> nextBatch() throws Exception;

    public abstract void reset() throws Exception;

    public boolean hasExpected() {
        return hasExpected;
    }

    public boolean hasNextBatch() {
        return hasNextBatch;
    }

    protected void incrementBatchCounter() {
        batchCounter++;
        if (batchCounter == numBatches)
            hasNextBatch = false;
    }

    protected void resetBatchCounter() {
        batchCounter = 0;
        hasNextBatch = true;
    }
    public int getBatchCounter(){
        return batchCounter;
    }
}
