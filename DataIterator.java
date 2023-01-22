import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

public class DataIterator {
    public int batchSize;
    public int numBatches;
    public double mean, standardDeviation;
    private int batchCounter, index;
    private boolean hasNextBatch;
    private DataPair[] dataPairs;

    public DataIterator(int batchSize, String dataPairsPath) throws Exception {
        this(batchSize, loadDataPairs(dataPairsPath));
    }

    public DataIterator(int batchSize, DataPair[] data) throws Exception {
        batchCounter = 0;
        hasNextBatch = true;
        index = 0;
        this.dataPairs = Stream.of(data).map(x -> x.clone()).toArray(DataPair[]::new);
        if (batchSize == 0) {
            this.batchSize = data.length;
        } else {
            this.batchSize = batchSize;
        }
        this.numBatches = data.length / this.batchSize;
        double[] meanAndStandardDeviation = StatisticalTesting.getMeanAndStandardDeviation(getData());
        this.mean = meanAndStandardDeviation[0];
        this.standardDeviation = meanAndStandardDeviation[1];
        for (DataPair dataPair : dataPairs) {
            dataPair.input.minus(this.mean).divide(this.standardDeviation).shape(1,
                    dataPair.input.getRows() * dataPair.input.getColumns()); // normalize and flatten
        }
    }

    public DataIterator() {

    }

    public Matrix[] getData() {
        return Stream.of(dataPairs).map(x -> x.input).toArray(Matrix[]::new);
    }

    public Matrix[] getLabels() {
        return Stream.of(dataPairs).map(x -> x.label).toArray(Matrix[]::new);
    }

    public void save(String saveFile) throws Exception {
        Map<String, Object> map = new HashMap<>();
        map.put("batchSize", batchSize);
        map.put("numBatches", numBatches);
        map.put("mean", mean);
        map.put("standardDeviation", standardDeviation);
        map.put("batchCounter", batchCounter);
        map.put("index", index);
        map.put("hasNextBatch", hasNextBatch);
        List<Map<String, Object>> dataPairsList = Stream.of(dataPairs).map(x -> x.save()).toList();
        map.put("dataPairs", dataPairsList);
        FileOutputStream fileOut = new FileOutputStream(saveFile);
        ObjectOutputStream out = new ObjectOutputStream(fileOut);
        out.writeObject(map);
        out.close();
        fileOut.close();
        return;
    }

    @SuppressWarnings("unchecked")
    public static DataIterator load(String saveFile) throws Exception {
        Map<String, Object> data = null;
        FileInputStream fileIn = new FileInputStream(saveFile);
        ObjectInputStream in = new ObjectInputStream(fileIn);
        data = (Map<String, Object>) in.readObject();
        in.close();
        fileIn.close();
        DataIterator output = new DataIterator();
        output.batchSize = (int) data.get("batchSize");
        output.numBatches = (int) data.get("numBatches");
        output.mean = (double) data.get("mean");
        output.standardDeviation = (double) data.get("standardDeviation");
        output.batchCounter = (int) data.get("batchCounter");
        output.index = (int) data.get("index");
        output.hasNextBatch = (boolean) data.get("hasNextBatch");
        List<Map<String, Object>> dataPairsList = (List<Map<String, Object>>) data.get("dataPairs");
        output.dataPairs = dataPairsList.stream().map(x -> DataPair.load(x)).toArray(DataPair[]::new);
        return output;
    }

    @SuppressWarnings("unchecked")
    private static DataPair[] loadDataPairs(String dataPairsPath) throws Exception {
        List<Map<String, Object>> temp;
        FileInputStream fileIn = new FileInputStream(dataPairsPath);
        ObjectInputStream in = new ObjectInputStream(fileIn);
        temp = (List<Map<String, Object>>) in.readObject();
        in.close();
        fileIn.close();
        Collections.shuffle(temp);
        return temp.stream().map(x -> DataPair.load(x)).toArray(DataPair[]::new);
    }

    public DataPair[] nextBatch() throws Exception {
        DataPair[] res = new DataPair[batchSize];
        for (int i = 0; i < batchSize; i++) {
            res[i] = dataPairs[index];
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
        DataPair output = dataPairs[index].clone();
        output.input.product(this.standardDeviation).add(this.mean);
        return output;
    }

    public DataPair[] get(int startIndex, int endIndex) {
        DataPair[] output = new DataPair[endIndex - startIndex];
        for (int i = startIndex; i < endIndex; i++) {
            output[i - startIndex] = dataPairs[i];
        }
        return output;
    }

}
