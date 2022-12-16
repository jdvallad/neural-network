import java.io.Serializable;
import java.util.List;

public abstract class DataIterator implements Serializable {
    protected final int batchSize;
    public final int numBatches;
    private int batchCounter;
    private boolean hasNextBatch;

    public DataIterator(int batchSize, int numBatches) {
        this.batchSize = batchSize;
        this.numBatches = numBatches;
        batchCounter = 0;
        hasNextBatch = true;
    }

    public abstract List<DataPair> nextBatch() throws Exception;

    public abstract void reset() throws Exception;

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
